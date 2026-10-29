#!/usr/bin/env python3
"""Local Vosk microphone listener; replies through the console's shared speaker."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.parse import quote

from vosk import Model, KaldiRecognizer, SetLogLevel
from greeting_rules import GreetingGate, parse_voice_command, voice_route_active, navigation_grammar

ROOT = Path(__file__).resolve().parent


def request(path, data=None, token=None):
    headers = {'Content-Type': 'application/json', 'X-Control-Client': 'voice-greeting'}
    if token: headers['X-Control-Token'] = token
    req = Request('http://127.0.0.1:8080/api/'+path,
                  data=json.dumps(data).encode() if data is not None else None, headers=headers)
    with urlopen(req, timeout=2) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=ROOT/'models/vosk-model-small-cn-0.22')
    parser.add_argument('--device', default='hw:CARD=XFMDPV0018,DEV=0')
    args = parser.parse_args()
    SetLogLevel(-1); model = Model(str(args.model)); gate = GreetingGate()
    directory = ROOT/'logs'; directory.mkdir(exist_ok=True)
    status_path = directory/'voice_greeting.json'
    state = {'enabled': True, 'listening': False, 'model': 'vosk-model-small-cn-0.22',
             'device': args.device, 'last_text': '', 'responses': 0, 'navigation_commands':0,
             'last_reply':'','rooms':[],'error': None}
    running = True; capture = None; last_write = 0.; token = None; next_heartbeat = 0.

    def save(force=False):
        nonlocal last_write
        now = time.monotonic()
        if not force and now-last_write < 1: return
        state['updated_at'] = time.time()
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, ensure_ascii=False)); os.replace(tmp, status_path)
        last_write = now

    def stop(*_):
        nonlocal running
        running = False
        if capture and capture.poll() is None: capture.terminate()

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        while running:
            recognizer = KaldiRecognizer(model, 16000); recognizer.SetWords(True)
            grammar_map = None
            try:
                capture = subprocess.Popen(['arecord', '-q', '-D', args.device, '-f', 'S16_LE',
                                            '-r', '16000', '-c', '1', '-t', 'raw'],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                state['error'] = None; next_status = 0.; speaker_busy = True
                was_listening = False
                while running:
                    data = capture.stdout.read(4000)
                    if not data:
                        error = capture.stderr.read().decode('utf-8', 'replace').strip()
                        raise RuntimeError(error or '麦克风采集已中断')
                    now = time.monotonic()
                    if now >= next_status:
                        try:
                            console = request('status'); speaker_busy = console['speech']['busy']
                            current_map=console.get('active_map')
                            if current_map!=grammar_map:
                                areas=request('areas?name='+quote(current_map))['areas'] if current_map else []
                                state['rooms']=[a['name'] for a in areas if a.get('target')]
                                grammar=navigation_grammar(state['rooms'])
                                recognizer=KaldiRecognizer(model,16000,json.dumps(grammar,ensure_ascii=False))
                                recognizer.SetWords(True);grammar_map=current_map;was_listening=False
                            if voice_route_active(console) and now >= next_heartbeat:
                                if token is None: token = request('session')['token']
                                try: request('heartbeat', {}, token)
                                except HTTPError as exc:
                                    if exc.code != 403: raise
                                    token = request('session')['token']; request('heartbeat', {}, token)
                                next_heartbeat = now+.7
                            state['error'] = None
                        except Exception as exc:
                            speaker_busy = True; state['error'] = '控制台未连接：'+str(exc)[:160]
                        next_status = now+.35
                    listening = gate.listening(now, speaker_busy)
                    state['listening'] = listening
                    if not listening:
                        recognizer.Reset(); was_listening = False; save(); continue
                    if not was_listening: recognizer.Reset()
                    was_listening = True
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result()); text = result.get('text', '')
                        if text and any(word in text for word in ('你好','客厅','厨房','卧室','书房')):
                            state['last_candidate']=text
                            words=result.get('result',[])
                            state['last_candidate_confidence']=round(sum(w.get('conf',0.) for w in words)/len(words),3) if words else 0.
                        # Do not log unrelated speech or retain raw audio.
                        command = parse_voice_command(text)
                        if command:
                            words = result.get('result', [])
                            confidence = sum(w.get('conf', 0.) for w in words)/len(words) if words else 0.
                            threshold = .80 if command['kind']=='navigate' else .65
                            if confidence >= threshold:
                                state['last_text'] = text; state['last_recognized_at'] = time.time()
                                token = request('session')['token']
                                response = request('voice/command', {'text': text}, token)
                                state['last_reply'] = response.get('reply', '你好' if response.get('accepted') else response.get('reason',''))
                                if response.get('accepted'):
                                    state['responses'] += 1; state['last_response_at'] = time.time()
                                    if command['kind']=='navigate':state['navigation_commands']+=1
                                    print(json.dumps({'heard': text, 'response': state['last_reply'],
                                                      'confidence': round(confidence, 3)}, ensure_ascii=False), flush=True)
                                else:
                                    print(json.dumps({'heard':text,'rejected':state['last_reply'],
                                                      'confidence':round(confidence,3)},ensure_ascii=False),flush=True)
                                gate.responded(now); recognizer.Reset(); was_listening = False
                    save()
            except Exception as exc:
                if running:
                    state['error'] = str(exc)[:240]; state['listening'] = False; save(True)
                    print(json.dumps({'error': state['error']}, ensure_ascii=False), flush=True)
            finally:
                if capture and capture.poll() is None:
                    capture.terminate()
                    try: capture.wait(timeout=2)
                    except subprocess.TimeoutExpired: capture.kill(); capture.wait()
            if running: time.sleep(3)
    finally:
        state['enabled'] = False; state['listening'] = False; save(True)


if __name__ == '__main__': main()

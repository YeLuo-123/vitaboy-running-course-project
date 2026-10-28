"""Nonblocking offline Mandarin speech through the robot's USB sound card."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading


class Speaker:
    def __init__(self, device=None):
        self.device=device or os.environ.get('X5_AUDIO_DEVICE','plughw:CARD=Device,DEV=0')
        self.lock=threading.Lock()
        self.generation=0
        self.process=None
        self.active=False
        self.text=''
        self.error=None

    def available(self):
        return shutil.which('espeak-ng') is not None and shutil.which('aplay') is not None

    def busy(self):
        with self.lock: return self.active

    def status(self):
        with self.lock:
            return {'available':self.available(),'busy':self.active,'text':self.text,'error':self.error}

    def cancel(self):
        with self.lock:
            self.generation+=1
            process=self.process
            self.process=None
            self.active=False
        if process and process.poll() is None:
            process.terminate()

    def say(self, text):
        if not self.available(): raise RuntimeError('离线语音程序未安装')
        if not isinstance(text,str) or not 1<=len(text)<=120:
            raise ValueError('播报文字长度无效')
        self.cancel()
        with self.lock:
            generation=self.generation
            self.active=True;self.text=text;self.error=None
        threading.Thread(target=self._run,args=(generation,text),daemon=True).start()

    def _execute(self, args, generation):
        with self.lock:
            if generation!=self.generation: return
            environment=None
            if args[:4] == ['aplay','-q','-D','pulse']:
                environment=os.environ.copy()
                environment.setdefault('PULSE_SERVER',f'unix:/run/user/{os.getuid()}/pulse/native')
                environment.setdefault('PULSE_SINK','alsa_output.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.analog-stereo')
            process=subprocess.Popen(args,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,env=environment)
            self.process=process
        try:
            _,error=process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill();process.communicate()
            raise RuntimeError('语音播放超时')
        finally:
            with self.lock:
                if self.process is process:self.process=None
        if generation==self.generation and process.returncode:
            raise RuntimeError(error.decode('utf-8','replace').strip()[:180] or '语音播放失败')

    def _run(self, generation, text):
        fd,path=tempfile.mkstemp(prefix='x5_voice_',suffix='.wav')
        os.close(fd)
        try:
            self._execute(['espeak-ng','-v','zh','-s','150','-w',path,text],generation)
            try:
                self._execute(['aplay','-q','-D',self.device,path],generation)
            except RuntimeError as exc:
                if 'Device or resource busy' not in str(exc) or self.device == 'pulse': raise
                self._execute(['aplay','-q','-D','pulse',path],generation)
        except Exception as exc:
            with self.lock:
                if generation==self.generation:self.error=str(exc)
        finally:
            Path(path).unlink(missing_ok=True)
            with self.lock:
                if generation==self.generation:self.active=False

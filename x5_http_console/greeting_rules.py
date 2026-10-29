"""Conservative, complete-utterance greeting matching and echo cooldown."""
import re


def parse_voice_command(text):
    if not isinstance(text, str) or len(text) > 60:
        return None
    normalized = re.sub(r'[\s，。！？、,.!?]', '', text)
    if is_greeting(normalized): return {'kind': 'greeting'}
    match = re.fullmatch(r'(?:小车|机器人)?你好(?:请)?去([\u4e00-\u9fffA-Za-z0-9_-]{1,24})', normalized)
    if match: return {'kind': 'navigate', 'room': match.group(1)}
    return None


def voice_route_active(console):
    route = console.get('route', {})
    return route.get('source') == 'voice' and route.get('phase') in ('announcing', 'running', 'dwell')


def navigation_grammar(rooms):
    phrases = ['你好', '你好 你好', '小车 你好', '机器人 你好', '[unk]']
    for room in rooms:
        if not isinstance(room,str) or not 1<=len(room)<=24:continue
        for prefix in ('你好 去 ', '你好 请 去 ', '小车 你好 去 ', '机器人 你好 去 '):
            phrases.append(prefix+room)
    return list(dict.fromkeys(phrases))


def is_greeting(text):
    if not isinstance(text, str) or len(text) > 60:
        return False
    normalized = re.sub(r'[\s，。！？、,.!?]', '', text)
    return normalized in ('你好', '你好你好', '你好你好你好', '小车你好', '机器人你好')


class GreetingGate:
    def __init__(self, cooldown=6., echo_tail=1.5):
        self.cooldown = cooldown; self.echo_tail = echo_tail
        self.block_until = 0.

    def listening(self, now, speaker_busy):
        if speaker_busy:
            self.block_until = max(self.block_until, now+self.echo_tail)
        return not speaker_busy and now >= self.block_until

    def responded(self, now):
        self.block_until = max(self.block_until, now+self.cooldown)

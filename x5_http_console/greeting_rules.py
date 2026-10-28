"""Conservative, complete-utterance greeting matching and echo cooldown."""
import re


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

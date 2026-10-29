import unittest
from unittest.mock import Mock
from greeting_rules import GreetingGate, is_greeting, parse_voice_command, voice_route_active, navigation_grammar
from speech import Speaker


class GreetingTests(unittest.TestCase):
    def test_only_complete_greetings_match(self):
        for text in ('你好', '你 好', '你好，你好！', '小车你好', '机器人你好'):
            self.assertTrue(is_greeting(text), text)
        for text in ('你好吗', '不要说你好', '他说你好之后离开了', '', None, 'hello'):
            self.assertFalse(is_greeting(text), text)

    def test_playback_echo_and_cooldown_suppress_recognition(self):
        gate = GreetingGate()
        self.assertTrue(gate.listening(10., False))
        gate.responded(10.)
        self.assertFalse(gate.listening(15., False))
        self.assertTrue(gate.listening(16., False))
        self.assertFalse(gate.listening(20., True))
        self.assertFalse(gate.listening(21., False))
        self.assertTrue(gate.listening(21.6, False))

    def test_navigation_requires_wake_phrase_and_exact_room_string(self):
        for room in ('客厅','厨房','卧室','书房'):
            self.assertEqual(parse_voice_command('你好，去'+room),{'kind':'navigate','room':room})
            self.assertEqual(parse_voice_command('你 好 请 去 '+room),{'kind':'navigate','room':room})
        for text in ('去客厅','你好别去客厅','不要你好去厨房','你好去','你好？我不要去客厅'):
            self.assertIsNone(parse_voice_command(text),text)
        self.assertEqual(parse_voice_command('你好去客厅和厨房')['room'],'客厅和厨房')

    def test_only_active_voice_routes_receive_listener_heartbeat(self):
        self.assertTrue(voice_route_active({'route':{'source':'voice','phase':'announcing'}}))
        self.assertTrue(voice_route_active({'route':{'source':'voice','phase':'running'}}))
        self.assertFalse(voice_route_active({'route':{'source':'web','phase':'running'}}))
        self.assertFalse(voice_route_active({'route':{'source':'voice','phase':'complete'}}))

    def test_grammar_includes_unknown_and_all_saved_rooms(self):
        grammar=navigation_grammar(['客厅','厨房','卧室','书房'])
        self.assertIn('[unk]',grammar)
        for room in ('客厅','厨房','卧室','书房'):self.assertIn('你好 去 '+room,grammar)

    def test_busy_sound_card_uses_shared_usb_output(self):
        speaker = Speaker(device='plughw:CARD=Device,DEV=0')
        speaker._execute = Mock(side_effect=[None, RuntimeError('Device or resource busy'), None])
        speaker._run(0, '你好')
        self.assertIsNone(speaker.error)
        self.assertEqual(speaker._execute.call_args_list[2].args[0][:4], ['aplay','-q','-D','pulse'])

    def test_other_playback_errors_are_reported(self):
        speaker = Speaker(); speaker._execute = Mock(side_effect=[None, RuntimeError('sound card missing')])
        speaker._run(0, '你好')
        self.assertEqual(speaker.error, 'sound card missing')
        self.assertEqual(speaker._execute.call_count, 2)


if __name__ == '__main__': unittest.main()

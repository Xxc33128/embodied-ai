import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

class BridgeContract(unittest.TestCase):
    def test_frozen_standard_is_loaded(self):
        import run_subagent_trials as mod
        self.assertTrue(hasattr(mod,'load_standard'),'frozen standard loader is missing')
        protocol,system,task=mod.load_standard()
        self.assertEqual(protocol['environment']['max_steps'],900)
        self.assertEqual(task,'Pick up the red block from the table and place it inside the blue bowl.')
        self.assertIn('900 robot control steps',system)
        self.assertNotIn('roughly x=+0.16',system)
    def test_policy_packet_preserves_information(self):
        path=Path(__file__).with_name('run_subagent_trials.py')
        self.assertTrue(path.exists(), 'policy transport bridge is not implemented')
        spec=importlib.util.spec_from_file_location('bridge',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        image=b'unchanged-png-bytes'
        raw={'model':'gpt-6-astra','messages':[{'role':'system','content':'exact instructions'},
             {'role':'user','content':[{'type':'text','text':'state[eef_state]: [1,2,3]'},
              {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image).decode()}}]}],
             'tools':[{'type':'function','function':{'name':'move_to','parameters':{'type':'object'}}}]}
        with tempfile.TemporaryDirectory() as t:
            packet=mod.export_request(raw,Path(t),1)
            self.assertEqual(packet['messages'][0],raw['messages'][0])
            self.assertEqual(packet['messages'][1]['content'][0],raw['messages'][1]['content'][0])
            self.assertEqual(packet['tools'],raw['tools'])
            block=packet['messages'][1]['content'][1]
            self.assertEqual(Path(block['image_url']['url']).read_bytes(),image)
            self.assertEqual(raw['messages'][1]['content'][1]['image_url']['url'][:5],'data:')
            self.assertNotIn('cube_pos',json.dumps(packet))
    def test_response_preserves_tool_arguments_without_invented_usage(self):
        import run_subagent_trials as mod
        chosen={'name':'move_to','arguments':{'targets':{'x':.1},'note':'observed'}}
        response=mod.chat_response(chosen,1)
        self.assertEqual(json.loads(response['choices'][0]['message']['tool_calls'][0]['function']['arguments']),chosen['arguments'])
        self.assertNotIn('usage',response)

if __name__=='__main__':unittest.main()

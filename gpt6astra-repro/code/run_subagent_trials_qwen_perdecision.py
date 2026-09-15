"""Qwen per-decision batch runner. Byte-identical protocol to run_subagent_trials.py; only the
decision-model identity labels differ (qwen3.8-flash-per-decision-subagent): one fresh
clean-room worker per model call, each loaded only with the current canonical request. Prompts, budgets,
recording, and frozen-bank checks are unchanged and still enforced against bowl-eval-v1.
Relay unchanged LLMAgentPolicy requests to a clean-room decision worker.

The real eval/agent/tool/action path is retained. No model decisions are generated
here. Public mailbox exports only the outgoing model request; private observations,
scoring, and videos stay in a separate experiment directory.
"""
import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
# The bowl-eval-v1 standard package lives in the embodied-ai repo and has been relocated more than
# once (breaking hard-coded paths and killing new runners with FileNotFoundError). Resolve robustly:
# explicit env var first, then a batch-local frozen copy next to this file, then both repo locations.
# Protocol content is still hash-verified by load_standard() (prompt SHA256 + assembled system SHA256 +
# environment source SHA256), so pointing at a byte-identical copy cannot silently change the protocol.
def _find_standard():
    cands=([Path(os.environ['BOWL_EVAL_STANDARD'])] if os.environ.get('BOWL_EVAL_STANDARD') else [])
    cands+=[ROOT/'standard', ROOT.parent/'embodied-ai'/'weeks'/'2026_0914-0920_GPT6Astra评测复现'/'bowl-eval-v1']
    for c in cands:
        if (c/'protocol.json').exists(): return c
    raise FileNotFoundError('bowl-eval-v1 standard dir not found; checked: '+', '.join(str(c) for c in cands))
STANDARD=_find_standard()

def load_standard():
    p=json.loads((STANDARD/'protocol.json').read_text())
    texts={}
    for key,rel in p['prompt_files'].items():
        raw=(STANDARD/rel).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=p['prompt_sha256'][key]:raise ValueError('prompt hash changed')
        texts[key]=raw.decode().strip()
    for name,sha in p['reference_source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('environment source changed: '+name)
    system=texts['system']+'\n\nRobot documentation:\n'+texts['embodiment']
    if hashlib.sha256(system.encode()).hexdigest()!=p['assembled_system_sha256']:raise ValueError('system assembly mismatch')
    return p,system,texts['task']

def snapshot(emb,obs,seed):
    env=emb._env;data=env.sim.data;model=env.sim.model
    return {'derived_seed':seed,'qpos':data.qpos.tolist(),'qvel':data.qvel.tolist(),
        'act':data.act.tolist(),'ctrl':data.ctrl.tolist(),'sim_time':float(data.time),
        'cube_pos':emb._cube_pos().tolist(),'cube_quat_wxyz':data.body_xquat[env.cube_body_id].tolist(),
        'cube_geom_sizes':{n:model.geom_size[model.geom_name2id(n)].tolist() for n in env.cube.contact_geoms},
        'robot_state':{k:v.tolist() for k,v in obs.state.items()},
        'image_raw_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in obs.images.items()},
        'controller_reset':'fresh environment; seed reset; binary gripper latch open; default OSC initialization'}

def prepare_bank(path):
    from PIL import Image
    from inspect_robots.scene import Scene
    from inspect_robots.rollout import derive_seed
    from robosuite_bowl import RobosuiteBowlEmbodiment
    protocol,_,task=load_standard();path=Path(path);path.mkdir(parents=True,exist_ok=False)
    atomic_json(path/'protocol.json',protocol)
    for i,seed in enumerate(protocol['experiment_design']['formal_scene_seeds']):
        folder=path/f'trial-{i+1:02d}';folder.mkdir()
        emb=RobosuiteBowlEmbodiment()
        try:
            actual=derive_seed(0,seed,0)
            obs=emb.reset(Scene(id=f'seed-{seed}',instruction=task,init_seed=seed),seed=actual)
            atomic_json(folder/'initialization.json',snapshot(emb,obs,actual))
            for k,v in obs.images.items():Image.fromarray(v).save(folder/f'{k}.png')
        finally:emb.close()
        print('initialized',i+1,flush=True)


def atomic_json(path, data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def export_request(body, public, call):
    public.mkdir(parents=True,exist_ok=True)
    packet=copy.deepcopy(body)
    for mi,msg in enumerate(packet.get('messages',[])):
        content=msg.get('content')
        if not isinstance(content,list):continue
        for bi,block in enumerate(content):
            if block.get('type')!='image_url':continue
            url=block['image_url']['url']
            if not url.startswith('data:image/png;base64,'):
                raise ValueError('unexpected image transport')
            data=base64.b64decode(url.split(',',1)[1])
            name='image-'+hashlib.sha256(data).hexdigest()+'.png'
            target=public/name
            if not target.exists():target.write_bytes(data)
            block['image_url']['url']=str(target)
    atomic_json(public/f'request-{call:02d}.json',packet)
    return packet


def chat_response(choice,call):
    return {'id':f'chatcmpl-subagent-{call}','object':'chat.completion',
            'model':'qwen3.8-flash-per-decision-subagent',
            'choices':[{'index':0,'message':{'role':'assistant','content':None,
                'tool_calls':[{'id':f'call-subagent-{call}','type':'function',
                  'function':{'name':choice['name'],'arguments':json.dumps(choice['arguments'],ensure_ascii=False)}}]},
                'finish_reason':'tool_calls'}]}


def run_trial(args):
    import httpx
    import numpy as np
    from inspect_robots import eval
    from inspect_robots.scene import Scene
    from inspect_robots.scorer import success_at_end,episode_length
    from inspect_robots.task import Task,Epochs
    from inspect_robots_agent import LLMAgentPolicy
    from robosuite_bowl import RobosuiteBowlEmbodiment
    from PIL import Image

    protocol,system_text,task_text=load_standard()

    private=Path(args.private).resolve();public=Path(args.public).resolve()
    private.mkdir(parents=True,exist_ok=False)
    public.mkdir(parents=True,exist_ok=True)
    started=time.time()
    class Relay(httpx.BaseTransport):
        count=0
        def handle_request(self,request):
            self.count+=1
            call=self.count
            body=json.loads(request.content)
            if body['messages'][0]['content']!=system_text:raise ValueError('nonstandard model prompt')
            atomic_json(private/f'outgoing-{call:02d}.json',body)
            atomic_json(private/f'call-{call:02d}-started.json',{'wall_time':time.time(),'physics_steps':env.steps})
            export_request(body,public,call)
            atomic_json(public/'status.json',{'state':'awaiting_decision','call':call,
                         'request':str(public/f'request-{call:02d}.json')})
            response_path=public/f'response-{call:02d}.json'
            t=time.time()
            if args.smoke:
                atomic_json(response_path,{'name':'give_up','arguments':{'reason':'transport smoke check','hindsight':'none'}})
            while not response_path.exists():
                if time.time()-t>1800:raise TimeoutError('No decision received in 30 minutes')
                time.sleep(.1)
            choice=json.loads(response_path.read_text())
            if set(choice)!={'name','arguments'} or not isinstance(choice['arguments'],dict):
                raise ValueError('Expected exactly one {name, arguments} tool call')
            atomic_json(public/'status.json',{'state':'executing','call':call})
            payload=chat_response(choice,call)
            atomic_json(private/f'incoming-{call:02d}.json',payload)
            atomic_json(private/f'call-{call:02d}-ended.json',{'wall_time':time.time(),'wait_seconds':time.time()-t,'physics_steps':env.steps})
            return httpx.Response(200,json=payload,request=request)

    class RecordedEmbodiment(RobosuiteBowlEmbodiment):
        def __init__(self):
            super().__init__()
            self.video=None;self.frames=0;self.steps=0;self.audit=open(private/'physics.jsonl','w')
        def record(self,obs,action=None,result=None):
            if self.video is None:
                self.video=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
                  '-f','rawvideo','-pix_fmt','rgb24','-s','672x224','-r','20','-i','-',
                  '-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(private/'video.mp4')],stdin=subprocess.PIPE)
            self.video.stdin.write(np.hstack([obs.images[c] for c in ['front','side','wrist']]).tobytes())
            self.frames+=1
            # Privileged fields are used only for the saved audit, never the request.
            cube=self._cube_pos()
            cube_quat=self._env.sim.data.body_xquat[self._env.cube_body_id].copy()
            grasped=bool(self._env._check_grasp(gripper=self._env.robots[0].gripper,object_geoms=self._env.cube))
            grippers=self._env.robots[0].gripper
            gs=list(grippers.values()) if isinstance(grippers,dict) else [grippers]
            geoms=[n for g in gs for n in g.contact_geoms]
            touched=bool(self._env.check_contact(geoms,self._env.cube))
            cube_geom=self._env.sim.model.geom_name2id(self._env.cube.contact_geoms[0])
            half=self._env.sim.model.geom_size[cube_geom]
            rot=self._env.sim.data.body_xmat[self._env.cube_body_id].reshape(3,3)
            bottom=float(cube[2]-abs(rot[2]).dot(half))
            row={'step':self.steps,'wall_s':time.time()-started,'state':{k:np.asarray(v).tolist() for k,v in obs.state.items()},
                 'cube_pos':cube.tolist(),'cube_quat_wxyz':cube_quat.tolist(),
                 'grasped':grasped,'any_gripper_contact':touched,'cube_bottom_z':bottom,
                 'cube_velocity':self._env.sim.data.get_body_xvelp(self._env.sim.model.body_id2name(self._env.cube_body_id)).tolist(),
                 'bowl_xy':self._env.bowl_xy.tolist(),
                 'info':dict(result.info) if result else {},
                 'action':np.asarray(action.data).tolist() if action is not None else None}
            self.audit.write(json.dumps(row)+'\n');self.audit.flush()
        def reset(self,scene,*,seed=None):
            obs=super().reset(scene,seed=seed)
            snap=snapshot(self,obs,seed)
            atomic_json(private/'initialization.json',snap)
            if not args.smoke:
                expected=json.loads((Path(args.bank)/f'trial-{args.index+1:02d}'/'initialization.json').read_text())
                if snap!=expected:raise ValueError('initial state or camera mismatch with frozen seed bank')
            atomic_json(private/'initialization-check.json',{'matched':not args.smoke,'smoke':args.smoke})
            for k,v in obs.images.items():Image.fromarray(v).save(private/f'initial-{k}.png')
            self.record(obs)
            return obs
        def step(self,action):
            r=super().step(action);self.steps+=1;self.record(r.observation,action,r);return r
        def close(self):
            if self.video is not None:
                final=self._observe(self._env._get_observations())
                for k,v in final.images.items():Image.fromarray(v).save(private/f'final-{k}.png')
                self.video.stdin.close();code=self.video.wait(timeout=60);self.video=None
                if code:raise RuntimeError('video encoding failed')
            if not self.audit.closed:self.audit.close()
            super().close()
    relay=Relay()
    env=RecordedEmbodiment()
    class StandardPolicy(LLMAgentPolicy):
        def reset(self,scene):
            super().reset(scene)
            self._messages[0]['content']=system_text
    try:
        scene_seed=1099 if args.smoke else 2000+args.index
        atomic_json(private/'manifest.json',{'protocol':protocol,'scene_seed':scene_seed,'trial_index':args.index,
            'access_mode':'one fresh clean-room worker per model call; each worker sees only the current canonical request (full text history plus the latest 2 observations) and has no memory of earlier decision steps; SOP 3.4 strict API-equivalent subagent variant','no_extra_settle':True,
            'decision_model':'qwen3.8-flash-per-decision-subagent','decision_model_snapshot':'qwen-token-plan-cn/qwen3.8-flash; pi workflow agent, thinking=medium set by parent harness front-end (not a provider snapshot id)','effort_effect':'medium requested in wire and pi-side thinking level set to medium; provider-side effort control not exposed'})
        task=Task(name='robosuite-bowl-qwen38flash-perdecision',scenes=[Scene(id=f'seed-{scene_seed}',
            instruction=task_text,init_seed=scene_seed)],
            scorer=[success_at_end(),episode_length()],max_steps=900,epochs=Epochs(count=1,reducer='mean'))
        policy=StandardPolicy(model='qwen3.8-flash-per-decision-subagent',base_url='http://subagent-relay.invalid/v1',
            api_key_env='SUBAGENT_UNUSED_KEY',env={},transport=relay,wire='chat',
            max_llm_calls=20,max_speed_frac=.25,effort='medium',images='always',image_horizon=2,depth='render')
        (log,)=eval(task,policy,env,seed=0,log_dir=str(private/'harness'))
        outcome={'status':log.status,'metrics':dict(log.results.metrics),
                 'calls':relay.count,'physics_steps':env.steps,'frames':env.frames,
                 'wall_seconds':time.time()-started,'trial_index':args.index,
                 'source':'qwen3.8-flash fresh per-decision subagent via LLMAgentPolicy transport',
                 'usage_tokens':None,'cost':None}
        # Read the serialized harness log for the authoritative termination reason.
        saved=next((private/'harness').glob('*.json'))
        outcome['termination_reasons']=json.loads(saved.read_text())['samples'][0]['termination_reasons']
        atomic_json(private/'result.json',outcome)
        print(json.dumps(outcome),flush=True)
    except Exception as exc:
        atomic_json(private/'error.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        env.close()
        atomic_json(public/'status.json',{'state':'finished','call':relay.count})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--private');ap.add_argument('--public');ap.add_argument('--bank')
    ap.add_argument('--index',type=int);ap.add_argument('--smoke',action='store_true');ap.add_argument('--prepare-bank')
    args=ap.parse_args()
    if args.prepare_bank:prepare_bank(args.prepare_bank)
    else:run_trial(args)

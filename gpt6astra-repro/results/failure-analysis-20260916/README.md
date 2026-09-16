# 三模型失败机制复核：证据附件

权威结论见[三模型失败机制复核](../../../weeks/2026_0914-0920_GPT6Astra评测复现/2026-09-16-三模型失败机制复核.md)。2026-09-16新增事后分析，原实验/成绩/提示词保持不变。

- `trial-audit.csv`：60轮的既有阶段、物理接触/举升辅助统计和原审核依据。
- `call-audit.csv`：1,109次真实模型调用的目标、公开note、执行前后状态及私有物理核验。最后调用之后由框架预算终止产生的额外停止步不计入该动作。
- `source-hashes.json`：120份原始physics/result文件的SHA256。
- `*-evidence.png`：从8个trial原视频选出的24个原始帧，三相机顺序front/side/wrist，标签数字为物理步/视频帧索引。它们是重点案例检查，不是新一轮全量盲评。
- `tool-timing-probe.json`：同一初态、同一10cm平移目标，在四种夹爪参数下真实工具生成的chunk步数。

私有物理字段只用于事后分析，不得用作后续policy的定位/抓持答案。公开note是已提交的简短动作说明，不含内部推理正文。

在原本地数据仍存在时复算：

```bash
python audit_failures.py --workspace <工作区根目录> --output <新输出目录>
<实验目录>/.venv-robosuite/bin/python probe_tool_timing.py --project <实验目录> --output <新探针JSON>
```

第一条只读取原始文件并导出派生表；第二条只构造工具和虚拟实测状态，不创建仿真场景、不调用模型。无需读取.env。

计数解释：stage来自既有review；any_grasp表示轨迹曾出现接触组合，不能等同稳定抓起。first_lift定义为某步grasped=true且cube_bottom_z>0.82m，仅供动作时序分析。相机/坐标解释与失败机制的因果推断请以主文的证据等级和限定为准。

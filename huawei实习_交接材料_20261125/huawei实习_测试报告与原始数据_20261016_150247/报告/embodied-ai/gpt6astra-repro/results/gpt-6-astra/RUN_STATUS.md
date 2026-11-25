# 本批执行状态

用户授权：20次测试，每trial一个全新GPT-6 Astra / medium子agent，连续完成本trial；不带其他trial或父对话信息。

调度agent：persistent20_manager。输入为冻结seed 2000–2019，初态复用已校验bank。物理/提示词/900步/20次policy决策不变。

离线检查命令：`.venv-robosuite/bin/python audit_standard_trials.py logs-persistent-subagent20-20260915-113640`。会产生summary.csv、recording-audit.json、index.html以及每trial/frames图像索引。

匿名复核：prepare_blind_review.py生成去掉模型notes/自动成绩/身份的录像与轨迹，复制到/tmp/bowl-blind-review-20260915后交给blind_video_review。正式映射仅在blind-review-map.json，不交给评审。

当前已完成trial-01：自动成功，10次决策429控制步，记录检查PASS。首次盲评stage3，认为末帧仍反弹、最终静置证据不足；第二盲评blind_review_second在进行。不得无痕覆盖分歧或为改善分数追加物理步。trial-02运行中。最终需要将匿名review结果归档到对应trial并汇总；保持自动成功与审核阶段分开。

本批接入差异：每个子agent保留本trial曾看过的图片，wire每次只送最近2组，但无法让已有上下文忘记旧图；因此单列persistent-subagent组，不声称独立API严格等价。unknown usage/cost填null。

## 恢复执行

用户已要求继续。原trial02在第14个request等待1800秒后超时，status=error、13个已执行决策、486步，保留不计分。新attempt目录trial-02-attempt-02，初态matched=true；由resume_persistent20调度，policy_02单agent连续完成。之后继续03–20。离线audit现在按trial_index选首个status=success的有效attempt，error attempt列入infrastructure-attempts.json。

trial01两个独立评审均确认阶段3以及释放入碗，最终放置状态存在3/4边界。review.json保持stage_max=3、stage_upper_bound=4、boundary_pending=true、success_confirmed=null；不把证据不足直接写成模型失败。

## 最新：root直接调度

用户明确禁止中间manager。manager均已退出，从trial04开始由root直接spawn单个policy。trial04完成（14calls421steps自动成功），记录审核4/4通过；trial05进行中，agent=/root/policy_05，runner session=96179，public=/tmp/bowl-persistent20-20260915-113640/trial-05，初态匹配。接着按序直到20，不重跑1–4。

policy05的prompt与dispatch首条完全相同，仅mailbox路径替换为trial-05。下一轮同理，fork_turns=none/model=gpt-6-astra/reasoning_effort=medium。每trial完整结束后原样保存agent final并追加dispatch结束事件，poll runner确认退出、audit，再启动下一预定index。禁止代替policy给任何机器人动作。

审阅：trial03的blind_review_remaining遇额度中断，review03尚未完成；trial04匿名包已生成，复核后续继续，不阻碍模型测试。

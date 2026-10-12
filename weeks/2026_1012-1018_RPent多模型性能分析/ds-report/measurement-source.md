# 计时定义源码证据〔E1〕

仅包含计时与结构采集代码；没有凭据或原始输入正文。

## instrument/ds_http_hooks.py
SHA256: 393e11cb6eee62133f4e06e4bf97205181e7fd7a5101af41e8b69d4720afaa97

```python
247: def _scan(state: dict, chunk: bytes, now: int) -> None:
248:     state["chunk_count"] += 1
249:     state["bytes_in"] += len(chunk)
250:     if state["first_chunk_ns"] is None:
251:         state["first_chunk_ns"] = now
252:     if state["last_chunk_ns"] is None or now > state["last_chunk_ns"]:
253:         state["last_chunk_ns"] = now
254:     state["buf"] += chunk
255:     while b"\n" in state["buf"]:
256:         line, state["buf"] = state["buf"].split(b"\n", 1)
257:         line = line.strip()
258:         if not line.startswith(b"data:"):
259:             continue
260:         payload = line[5:].strip()
261:         if payload == b"[DONE]":
262:             continue
263:         try:
264:             obj = json.loads(payload)
265:         except Exception:  # noqa: BLE001
266:             continue
267:         state["events"] += 1
268:         if state["first_event_ns"] is None:
269:             state["first_event_ns"] = now
270:         if obj.get("model"):
271:             state["model_returned"] = obj["model"]
272:         if obj.get("usage"):
273:             state["usage"] = obj["usage"]
274:         for choice in obj.get("choices") or []:
275:             if not isinstance(choice, dict):
276:                 continue
277:             if choice.get("finish_reason"):
278:                 state["finish_reason"] = choice["finish_reason"]
279:             delta = choice.get("delta") or {}
280:             if delta.get("reasoning_content"):
281:                 state["reasoning_chars"] += len(delta["reasoning_content"])
282:                 if state["first_reasoning_ns"] is None:
283:                     state["first_reasoning_ns"] = now
284:             if delta.get("content"):
285:                 state["content_chars"] += len(delta["content"])
286:                 if state["first_content_ns"] is None:
287:                     state["first_content_ns"] = now
288:             if delta.get("tool_calls"):
289:                 state["tool_deltas"] += len(delta["tool_calls"])
290:                 if state["first_tool_ns"] is None:
291:                     state["first_tool_ns"] = now
292:
```

```python
323:
324: def _emit_response(state: dict) -> None:
325:     if state.get("emitted"):
326:         return
327:     state["emitted"] = True
328:     info = state["info"]
329:     start_ns = state["start_ns"]
330:     row = _new_row("llm.response")
331:     row.update({
332:         "start_ns": start_ns,
333:         "end_ns": state["last_chunk_ns"] or time.monotonic_ns(),
334:         "logical_request_id": info.get("logical_request_id"),
335:         "attempt": info.get("attempt"),
336:         "url_path": info.get("url_path"),
337:         "body_sha256": info.get("body_sha256"),
338:         "hdr_ms": _ms(start_ns, state["hdr_ns"]),
339:         "first_chunk_ms": _ms(start_ns, state["first_chunk_ns"]),
340:         "first_event_ms": _ms(start_ns, state["first_event_ns"]),
341:         "first_reasoning_ms": _ms(start_ns, state["first_reasoning_ns"]),
342:         "first_content_ms": _ms(start_ns, state["first_content_ns"]),
343:         "first_tool_ms": _ms(start_ns, state["first_tool_ns"]),
344:         "last_chunk_ms": _ms(start_ns, state["last_chunk_ns"]),
345:         "thinking_span_ms": (None if state["first_reasoning_ns"] is None or state["first_content_ns"] is None
346:                              else _ms(state["first_reasoning_ns"], state["first_content_ns"])),
347:         "output_span_ms": (None if state["first_content_ns"] is None or state["last_chunk_ns"] is None
348:                            else _ms(state["first_content_ns"], state["last_chunk_ns"])),
349:         "events": state["events"],
350:         "chunk_count": state["chunk_count"],
```

```python
366:         pass
367:
368:
369: def _emit_transport(state: dict, status_code=None, headers=None, error=None) -> None:
370:     info = state["info"]
371:     start_ns = state["start_ns"]
372:     row = _new_row("llm.transport")
373:     row.update({
374:         "start_ns": start_ns,
375:         "end_ns": state["hdr_ns"] or time.monotonic_ns(),
376:         "logical_request_id": info.get("logical_request_id"),
377:         "attempt": info.get("attempt"),
378:         "url_path": info.get("url_path"),
379:         "method": info.get("method"),
380:         "body_sha256": info.get("body_sha256"),
381:         "body_bytes": info.get("body_bytes"),
382:         "hdr_ms": _ms(start_ns, state["hdr_ns"]),
383:         "status_code": status_code,
384:         "request_params": {k: v for k, v in info.items() if k.startswith("param_")},
385:         "param_keys": info.get("param_keys"),
386:         "n_messages": info.get("n_messages"),
387:         "n_images": info.get("n_images"),
388:         "image_approx_bytes": info.get("image_approx_bytes"),
389:         "n_tools": info.get("n_tools"),
390:         "tools_sha256": info.get("tools_sha256"),
391:     })
392:     if headers:
393:         row["response_headers"] = headers
394:     if error:
395:         row["status"] = "error"
396:         row["error_type"] = error
397:     row["duration_ns"] = row["end_ns"] - row["start_ns"]
```

## instrument/timing_hooks.py
SHA256: c5ae6c03636f2a2c6e3d5dd649718ffc28e8004a08d8e448c0bb1b004ef67b88

```python
145: def _shape(value):
146:     if value is None: return {'type':'none'}
147:     out={'type':type(value).__name__}
148:     if hasattr(value,'shape'): out['shape']=[int(n) for n in value.shape]
149:     if hasattr(value,'dtype'): out['dtype']=str(value.dtype)
150:     if hasattr(value,'device'): out['device']=str(value.device)
151:     return out
152:
153:
154: def _text(value):
155:     if not isinstance(value,str): return {'type':type(value).__name__}
156:     return {'chars':len(value),'utf8_bytes':len(value.encode()),'sha256':hashlib.sha256(value.encode()).hexdigest()}
157:
158:
159: def _arg(args,kwargs,index,key):
160:     return kwargs[key] if key in kwargs else (args[index] if len(args)>index else None)
161:
162:
163: def _metadata_input(row,args,kwargs):
164:     name=row['name']
165:     if name not in {'pi.handler','sam.handler','sam.client.total','pi.input_transform'}: return
166:     t=time.monotonic_ns()
167:     try:
168:         if name=='pi.handler':
169:             obs=_arg(args,kwargs,1,'obs')
170:             if isinstance(obs,dict):
171:                 row['input_meta']={k:([_text(v) for v in val] if k=='task_descriptions' and isinstance(val,(list,tuple)) else _shape(val)) for k,val in obs.items() if k in {'main_images','wrist_images','extra_view_images','states','task_descriptions'}}
172:         elif name in {'sam.handler','sam.client.total'}:
173:             meta={};row['input_meta']=meta
174:             if name=='sam.handler':
175:                 encoded=_arg(args,kwargs,1,'image_base64')
176:                 data=base64.b64decode(encoded,validate=True)
177:                 meta['image_base64_chars']=len(encoded)
178:             else:
179:                 image=_arg(args,kwargs,1,'image')
180:                 if isinstance(image,(bytes,bytearray,memoryview)):
181:                     data=bytes(image)
182:                 else:
183:                     meta['image_array']=_shape(image)
184:                     meta['fingerprint_unavailable']='ndarray-not-encoded-yet'
185:                     return  # Actual tools.py passes image artifact bytes.
186:             meta['image_bytes']=len(data)
187:             meta['image_sha256']=hashlib.sha256(data).hexdigest()
188:             text=kwargs.get('text_prompt')
189:             text=text.strip() if isinstance(text,str) else None
190:             point=kwargs.get('point')
191:             point=[int(x) for x in point] if point is not None else None
192:             if text is not None: meta['text_prompt']=_text(text)
193:             if point is not None:
194:                 meta['point']={'length':len(point),'sha256':hashlib.sha256(json.dumps(point,separators=(',',':')).encode()).hexdigest()}
195:             fingerprint={'image_sha256':meta['image_sha256'],'text':text,'point':point,'min_score':float(kwargs.get('min_score',.2))}
196:             meta['request_fingerprint']=hashlib.sha256(json.dumps(fingerprint,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
197:             image_module=sys.modules.get('PIL.Image')
198:             if image_module is not None:
199:                 with image_module.open(io.BytesIO(data)) as image:
200:                     meta['image_hw']=[int(image.height),int(image.width)]
201:     except Exception as exc:
202:         row['metadata_input_error']=type(exc).__name__
203:     finally: row['metadata_input_ns']=time.monotonic_ns()-t
204:
205:
206: def _metadata_output(row,result):
207:     name=row['name']
208:     if name not in {'pi.handler','sam.handler','sam.client.total','pi.input_transform'}: return
209:     t=time.monotonic_ns()
210:     try:
211:         if name=='pi.handler': row['output_meta']=_shape(result)
212:         elif name=='pi.input_transform' and isinstance(result,dict):
213:             meta={k:_shape(result[k]) for k in ('tokenized_prompt','tokenized_prompt_mask','state') if k in result}
214:             row['output_meta']=meta
215:             mask=result.get('tokenized_prompt_mask')
216:             if mask is not None and os.environ.get('RPENT_TIMING_MASK_COUNT','1') != '0':
217:                 device=str(getattr(mask,'device','cpu'))
218:                 device_sync=device.partition(':')[0] not in ('cpu','')
219:                 row['metadata_device_sync']=device_sync
220:                 if device_sync: row['sync_disturbed']=True
221:                 count=mask.sum(); meta['tokenized_prompt_mask']['nonpadding_count']=int(count.item() if hasattr(count,'item') else count)
222:         elif name=='sam.handler' and isinstance(result,dict):
223:             row['output_meta']={'keys':sorted(k for k in result if k in {'found','reason','score','box','mask_png_base64','mask_shape'})}
224:             if isinstance(result.get('mask_shape'),(list,tuple)): row['output_meta']['mask_shape']=[int(x) for x in result['mask_shape']]
225:             if isinstance(result.get('found'),bool): row['output_meta']['found']=result['found']
```

## extract_metrics.py
SHA256: ec43a9c5ab40397217a931d2fc7731f15bda1584c32b76cd08f49b73e1e5bc8a

```python
263:             if row["name"] != "llm.response":
264:                 continue
265:             usage = row.get("usage") or {}
266:             details = usage.get("completion_tokens_details") or {}
267:             prompt_details = usage.get("prompt_tokens_details") or {}
268:             # Phase markers only exist when the wire actually streamed. In the
269:             # formal 10/09 batch (non-streaming requests) these are all None --
270:             # that is the honest value, not a gap to be filled by a guess. In the
271:             # B2 shim batch they carry real per-token timings.
272:             deltas = [v for v in (row.get("first_reasoning_ms"), row.get("first_content_ms"),
273:                                   row.get("first_tool_ms")) if v is not None]
274:             ttft = min(deltas) if deltas else None
275:             last = row.get("last_chunk_ms")
276:             tokens = usage.get("completion_tokens")
277:             thinking = (row["first_content_ms"] - row["first_reasoning_ms"]
278:                         if row.get("first_content_ms") is not None
279:                         and row.get("first_reasoning_ms") is not None else None)
280:             tpot = ((last - ttft) / (tokens - 1)
281:                     if ttft is not None and last is not None and tokens and tokens > 1 else None)
282:             llm_req.append({
283:                 "logical_request_id": row.get("logical_request_id"),
284:                 "attempt": row.get("attempt"),
285:                 "duration_ms": round(row["duration_ns"] / 1e6, 3),
286:                 "first_byte_ms": row.get("first_chunk_ms"),
287:                 "server_span_ms": (round((row.get("last_chunk_ms") or 0) - (row.get("hdr_ms") or 0), 3)
288:                                    if row.get("last_chunk_ms") is not None and row.get("hdr_ms") is not None else None),
289:                 "prompt_tokens": usage.get("prompt_tokens"),
290:                 "cached_tokens": prompt_details.get("cached_tokens"),
291:                 "completion_tokens": usage.get("completion_tokens"),
292:                 "reasoning_tokens": details.get("reasoning_tokens"),
293:                 "finish_reason": row.get("finish_reason"),
294:                 "model_returned": row.get("model_returned"),
295:                 "body_sha256": (row.get("body_sha256") or "")[:16],
296:                 "streamed": bool(row.get("stream")),
297:                 "ttft_ms": round(ttft, 3) if ttft is not None else None,
298:                 "first_reasoning_ms": row.get("first_reasoning_ms"),
299:                 "first_content_ms": row.get("first_content_ms"),
300:                 "first_tool_ms": row.get("first_tool_ms"),
301:                 "last_chunk_ms": last,
302:                 "thinking_visible_ms": round(thinking, 3) if thinking is not None else None,
303:                 "tpot_ms": round(tpot, 3) if tpot is not None else None,
304:                 "chunk_count": row.get("chunk_count"),
305:             })
```

```python
334:             "result": {
335:                 "process_exit_code": parse_process_status(run_dir / "process-status.txt"),
336:                 "first_log_ts": run_log["first_ts"],
337:                 "last_log_ts": run_log["last_ts"],
338:                 "turns": run_log["turns"],
339:                 "finish_status": run_log["finish_status"],
340:                 "solved": run_log["finish_status"] == "success",
341:                 "recipe_file": run_log["recipe_file"],
342:                 "rpent_elapsed_s": run_log["rpent_reported"]["elapsed_s"],
343:                 "terminated_true_mentions": run_log["terminated_true_mentions"],
```
## robots/libero/prompts/evaluate.py
SHA256: 9efbc638d40043d183ac46898182b55bf057c36fd0fd5f6b469f406ba748a50b

```python
571: def system_prompt(
572:     variables: Mapping[str, object] | None = None,
573: ) -> PromptNode:
574:     """Assemble the LIBERO evaluation prompt for the selected memory profile."""
575:     if (variables or {}).get("memory_profile", "hf") == "local":
576:         return {
577:             "ROLE AND EVALUATION": ROLE_AND_EVALUATION,
578:             "MEMORY PROFILE — LOCAL SUITE + TASK + GLOBAL": (LOCAL_MEMORY_PROFILE),
579:             "PROVEN LEVERS": PROVEN_LEVERS,
580:             "RUNTIME": RUNTIME,
581:             "YOUR GOAL": GOAL,
582:             "RULES (NON-NEGOTIABLE)": RULES,
583:             "LOCALIZATION": LOCALIZATION,
584:             "FIRST-STEP ALGORITHM": PERCEPTION_ALGORITHM,
585:             "WORKFLOW": Numbered(LOCAL_WORKFLOW_STEPS),
586:             "KEY HYPERPARAMETERS": KEY_HYPERPARAMETERS,
587:             "OUTPUT DISCIPLINE": OUTPUT_DISCIPLINE,
588:         }
```

## robots/libero/prompt_bundle.py
SHA256: f29ea4a268a4f155b6ec5f57ffbf32864e2c003da81c2c173fa48174b1feffce

```python
27: def system_prompt(
28:     variables: Mapping[str, object] | None = None,
29: ) -> PromptNode:
30:     """Assemble the LIBERO system prompt for the selected run mode."""
31:     if (variables or {}).get("mode", "eval") == "explore":
32:         return explore_parts.system_prompt()
33:     return evaluate_parts.system_prompt(variables)
```

## rpent/cli/main.py
SHA256: 562c0fe4b6e112e7a6334df27845359749f06a4946f7a4113b30ec3a3181494f

```python
460:     prompt_bundle = robot_spec.prompts
461:     prompt_vars = {**prompt_vars, "output_dir": output_dir}
462:     system_prompt = prompt_bundle.render(
463:         "system",
464:         variables=prompt_vars,
465:     )
466:     user_msg = prompt_bundle.render(
467:         "user",
468:         variables=prompt_vars,
469:     )
```

## rpent/planner/api_loop.py
SHA256: 42a1dd1cfee83ef03b491d37ea7cd445b205bd60f1d066adfa4acd29070d0bf4

```python
362:     def _build_agent(self, system_prompt: str, toolkit: Toolkit) -> Agent:
363:         """Build an Agent for terminal or Dashboard execution."""
364:         thinking_effort: str | bool = self._reasoning_effort
365:         if thinking_effort == "none":
366:             thinking_effort = False
367:         return Agent(
368:             self._model,
369:             instructions=system_prompt or None,
370:             tools=_build_tools(toolkit, no_images=self._no_images),
371:             model_settings=_build_model_settings(self._model, self._max_tokens),
372:             capabilities=[
373:                 Thinking(effort=thinking_effort),
374:                 ProcessHistory(processor=_prune_history_images),
375:             ],
```

/** 冒烟自检：harness + corpus + sqlite 介质 + stub LLM 全链路。 */
import { MemoryStore } from '../lib/memory-store.js'
import { createVectorIndex } from '../lib/vector-store.js'
import { createEmbeddingProvider } from '../lib/embedding.js'
import { benchConfig, memoryUnit, SqliteMedium, stubLlm, makeRecord } from './support/harness.mjs'
import { generateCorpus } from './support/corpus.mjs'

const cfg = benchConfig()
const { texts } = generateCorpus(8)
const emb = createEmbeddingProvider(cfg.embedding)
const vecs = await emb.embed(texts)

// memory 介质
const memStore = await MemoryStore.open(memoryUnit(), createVectorIndex(), cfg)
for (const [i, t] of texts.entries()) await memStore.putRecord(makeRecord({ text: t, embedding: vecs[i] }))
await memStore.refreshNeighborhood(memStore.allMemories()[0].id, 10)
const hits = memStore.similar(vecs[0], 3)
console.log('memory store: records=', memStore.allMemories().length, 'similar top cos=', hits[0]?.cos.toFixed(3))

// sqlite 介质
const medium = new SqliteMedium('smoke')
const unit = await medium.openUnit('dsh_mem_bench', cfg)
const sqStore = await MemoryStore.open(unit, createVectorIndex(), cfg)
await sqStore.putRecord(makeRecord({ text: texts[0], embedding: vecs[0] }))
await sqStore.close()
await medium.close()
const insp = await medium.inspect('dsh_mem_bench')
console.log('sqlite: rows=', insp.tables.episodic.rows, 'payloadBytes=', insp.tables.episodic.payloadBytes, 'fileBytes=', insp.fileBytes)
const afterVac = await medium.vacuum()
console.log('after VACUUM:', afterVac, 'bytes')
await medium.remove()

// stub LLM 各分支
const llm = stubLlm()
await llm.complete('You distill durable memories…', 'x')
await llm.complete('You score how strongly…', 'x')
await llm.complete('You judge how a NEW memory…', 'x')
await llm.complete('You distill the shared semantic core…', 'x')
await llm.complete('You write question-answer pairs…', 'x')
await llm.complete('You answer a question using ONLY the provided summary statement.', 'x')
await llm.complete('You judge whether a NEW distilled statement can be folded into an EXISTING semantic node.', 'x')
console.log('stub calls:', JSON.stringify(llm.calls))
console.log('SMOKE OK')

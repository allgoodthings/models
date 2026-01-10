#!/usr/bin/env node
/**
 * Test concurrent audio generation with Higgs Audio vLLM
 *
 * Usage: node test-concurrent.mjs [pod-url]
 * Example: node test-concurrent.mjs https://luncbsq07o13z0-8000.proxy.runpod.net
 */

import fs from 'fs';
import path from 'path';

const POD_URL = process.argv[2] || 'https://luncbsq07o13z0-8000.proxy.runpod.net';

const speeches = [
  {
    name: 'excited-announcement',
    system: `Generate audio following instruction.

<|scene_desc_start|>
An excited tech announcer at a product launch. Energetic, fast-paced, enthusiastic delivery with rising intonation.
<|scene_desc_end|>`,
    text: 'Hello and welcome to the future of audio generation! This is absolutely incredible technology that will change everything!'
  },
  {
    name: 'mlk-dream',
    system: `Generate audio following instruction.

<|scene_desc_start|>
A powerful civil rights speech. Deep, resonant voice with deliberate pacing. Pauses for emphasis. Builds from quiet determination to passionate crescendo.
<|scene_desc_end|>`,
    text: 'I have a dream that one day this nation will rise up and live out the true meaning of its creed. We hold these truths to be self-evident, that all men are created equal.'
  },
  {
    name: 'churchill-defiant',
    system: `Generate audio following instruction.

<|scene_desc_start|>
A wartime leader rallying the nation. British accent. Gravelly, defiant tone. Each phrase punched with conviction. Dramatic pauses between clauses.
<|scene_desc_end|>`,
    text: 'We shall fight on the beaches, we shall fight on the landing grounds, we shall fight in the fields and in the streets, we shall fight in the hills. We shall never surrender.'
  }
];

async function generateSpeech(speech) {
  const start = Date.now();
  console.log(`[${speech.name}] Starting request...`);

  const response = await fetch(`${POD_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'higgs-audio-v2',
      messages: [
        { role: 'system', content: speech.system },
        { role: 'user', content: speech.text }
      ],
      temperature: 0.5,
      max_tokens: 4096
    })
  });

  const elapsed = ((Date.now() - start) / 1000).toFixed(2);

  if (!response.ok) {
    const error = await response.text();
    console.error(`[${speech.name}] Failed (${elapsed}s): ${error}`);
    return null;
  }

  const data = await response.json();
  console.log(`[${speech.name}] Response received (${elapsed}s)`);

  // Extract audio from response
  const choice = data.choices?.[0];
  if (!choice) {
    console.error(`[${speech.name}] No choices in response`);
    return null;
  }

  // Check for audio in the message
  const message = choice.message;
  const audio = message?.audio;

  if (audio?.data) {
    // Audio is base64 encoded
    const audioBuffer = Buffer.from(audio.data, 'base64');
    const filename = `output-${speech.name}.${audio.format || 'wav'}`;
    fs.writeFileSync(filename, audioBuffer);
    console.log(`[${speech.name}] Saved: ${filename} (${(audioBuffer.length / 1024).toFixed(1)} KB)`);
    return { name: speech.name, file: filename, elapsed: parseFloat(elapsed) };
  } else {
    // Maybe audio is in content as text tokens?
    console.log(`[${speech.name}] Response structure:`, JSON.stringify(message, null, 2).slice(0, 500));
    return null;
  }
}

async function main() {
  console.log(`\nTesting concurrent audio generation`);
  console.log(`Pod URL: ${POD_URL}`);
  console.log(`Speeches: ${speeches.length}\n`);

  const totalStart = Date.now();

  // Fire all requests concurrently
  const results = await Promise.all(speeches.map(generateSpeech));

  const totalElapsed = ((Date.now() - totalStart) / 1000).toFixed(2);
  const successful = results.filter(r => r !== null);

  console.log(`\n${'='.repeat(50)}`);
  console.log(`Results:`);
  console.log(`  Total time: ${totalElapsed}s`);
  console.log(`  Successful: ${successful.length}/${speeches.length}`);

  if (successful.length > 0) {
    const avgTime = (successful.reduce((a, b) => a + b.elapsed, 0) / successful.length).toFixed(2);
    console.log(`  Avg per request: ${avgTime}s`);
    console.log(`  Speedup vs sequential: ${(successful.length * parseFloat(avgTime) / parseFloat(totalElapsed)).toFixed(2)}x`);
  }
  console.log(`${'='.repeat(50)}\n`);
}

main().catch(console.error);

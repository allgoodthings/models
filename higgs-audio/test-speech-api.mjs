#!/usr/bin/env node
/**
 * Test concurrent audio generation using /v1/audio/speech endpoint
 *
 * Usage: node test-speech-api.mjs [pod-url]
 * Example: node test-speech-api.mjs https://luncbsq07o13z0-8000.proxy.runpod.net
 */

import fs from 'fs';

const POD_URL = process.argv[2] || 'https://luncbsq07o13z0-8000.proxy.runpod.net';

const speeches = [
  {
    name: '01-short',
    text: 'Hello and welcome to the future of audio generation. This is a test of concurrent processing.'
  },
  {
    name: '02-mlk',
    text: 'I have a dream that one day this nation will rise up and live out the true meaning of its creed. We hold these truths to be self-evident, that all men are created equal.'
  },
  {
    name: '03-churchill',
    text: 'We shall fight on the beaches, we shall fight on the landing grounds, we shall fight in the fields and in the streets, we shall fight in the hills. We shall never surrender.'
  }
];

async function generateSpeech(speech) {
  const start = Date.now();
  console.log(`[${speech.name}] Starting request...`);

  const response = await fetch(`${POD_URL}/v1/audio/speech`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'higgs-audio-v2',
      input: speech.text,
      voice: 'belinda',
      response_format: 'wav'
    })
  });

  const elapsed = ((Date.now() - start) / 1000).toFixed(2);

  if (!response.ok) {
    const error = await response.text();
    console.error(`[${speech.name}] Failed (${elapsed}s): ${error.slice(0, 200)}`);
    return null;
  }

  const audioBuffer = Buffer.from(await response.arrayBuffer());
  const filename = `output-${speech.name}.wav`;
  fs.writeFileSync(filename, audioBuffer);
  console.log(`[${speech.name}] Saved: ${filename} (${(audioBuffer.length / 1024).toFixed(1)} KB) in ${elapsed}s`);

  return { name: speech.name, file: filename, elapsed: parseFloat(elapsed), size: audioBuffer.length };
}

async function main() {
  console.log(`\n${'='.repeat(50)}`);
  console.log(`Testing concurrent audio generation (/v1/audio/speech)`);
  console.log(`Pod URL: ${POD_URL}`);
  console.log(`Speeches: ${speeches.length}`);
  console.log(`${'='.repeat(50)}\n`);

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
    const totalSize = (successful.reduce((a, b) => a + b.size, 0) / 1024 / 1024).toFixed(2);
    console.log(`  Avg per request: ${avgTime}s`);
    console.log(`  Total audio: ${totalSize} MB`);

    if (successful.length > 1) {
      const sequentialTime = successful.reduce((a, b) => a + b.elapsed, 0);
      const speedup = (sequentialTime / parseFloat(totalElapsed)).toFixed(2);
      console.log(`  Speedup vs sequential: ${speedup}x`);
    }
  }
  console.log(`${'='.repeat(50)}\n`);
}

main().catch(console.error);

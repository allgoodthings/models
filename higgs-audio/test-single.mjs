#!/usr/bin/env node
/**
 * Test single audio generation
 */

import fs from 'fs';

const POD_URL = process.argv[2] || 'https://luncbsq07o13z0-8000.proxy.runpod.net';

async function main() {
  console.log(`Testing single request to ${POD_URL}`);

  const start = Date.now();

  const response = await fetch(`${POD_URL}/v1/audio/speech`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'higgs-audio-v2',
      input: 'Hello and welcome to the future of audio generation. This is a test.',
      voice: 'belinda',
      response_format: 'wav'
    })
  });

  const elapsed = ((Date.now() - start) / 1000).toFixed(2);

  if (!response.ok) {
    console.error(`Failed: ${await response.text()}`);
    return;
  }

  const audioBuffer = Buffer.from(await response.arrayBuffer());
  fs.writeFileSync('output-single.wav', audioBuffer);
  console.log(`Saved: output-single.wav (${(audioBuffer.length / 1024).toFixed(1)} KB) in ${elapsed}s`);
}

main().catch(console.error);

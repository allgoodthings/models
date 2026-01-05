#!/usr/bin/env node
/**
 * Test script for the lip-sync API.
 *
 * Usage:
 *   node test-api.mjs --video input.mp4 --audio speech.wav --output result.mp4 --bbox 100,50,200,250
 *   node test-api.mjs --video input.mp4 --audio speech.wav --config faces.json --output result.mp4
 *
 * Options:
 *   --video      Input video file (required)
 *   --audio      Audio file for lip-sync (required)
 *   --output     Output video file (default: output.mp4)
 *   --bbox       Face bounding box as x,y,w,h (for single face)
 *   --config     JSON file with face configuration (for multi-face)
 *   --start      Start time in ms (default: 0)
 *   --end        End time in ms (default: video duration)
 *   --api        API base URL (default: http://localhost:8000)
 *   --no-enhance Disable CodeFormer enhancement
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { basename } from 'path';

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    video: null,
    audio: null,
    output: 'output.mp4',
    bbox: null,
    config: null,
    start: 0,
    end: null,
    api: 'http://localhost:8000',
    enhance: true,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '--video':
        opts.video = args[++i];
        break;
      case '--audio':
        opts.audio = args[++i];
        break;
      case '--output':
        opts.output = args[++i];
        break;
      case '--bbox':
        opts.bbox = args[++i];
        break;
      case '--config':
        opts.config = args[++i];
        break;
      case '--start':
        opts.start = parseInt(args[++i], 10);
        break;
      case '--end':
        opts.end = parseInt(args[++i], 10);
        break;
      case '--api':
        opts.api = args[++i];
        break;
      case '--no-enhance':
        opts.enhance = false;
        break;
      case '--help':
      case '-h':
        console.log(`
Lip-Sync API Test Script

Usage:
  node test-api.mjs --video input.mp4 --audio speech.wav --bbox 100,50,200,250
  node test-api.mjs --video input.mp4 --audio speech.wav --config faces.json

Options:
  --video      Input video file (required)
  --audio      Audio file for lip-sync (required)
  --output     Output video file (default: output.mp4)
  --bbox       Face bounding box as x,y,w,h (for single face)
  --config     JSON file with face configuration (for multi-face)
  --start      Start time in ms (default: 0)
  --end        End time in ms (required if no --config)
  --api        API base URL (default: http://localhost:8000)
  --no-enhance Disable CodeFormer enhancement

Example faces.json:
{
  "faces": [
    {"character_id": "alice", "bbox": [100, 50, 200, 250], "start_time_ms": 0, "end_time_ms": 5000},
    {"character_id": "bob", "bbox": [400, 60, 180, 220], "start_time_ms": 1000, "end_time_ms": 5000}
  ]
}
`);
        process.exit(0);
    }
  }

  return opts;
}

// Build the request configuration
function buildRequestConfig(opts) {
  // If config file provided, use it
  if (opts.config) {
    if (!existsSync(opts.config)) {
      throw new Error(`Config file not found: ${opts.config}`);
    }
    const config = JSON.parse(readFileSync(opts.config, 'utf-8'));
    return {
      faces: config.faces,
      enhance_quality: opts.enhance,
      fidelity_weight: config.fidelity_weight ?? 0.7,
    };
  }

  // Otherwise build from bbox
  if (!opts.bbox) {
    throw new Error('Either --bbox or --config is required');
  }

  const bboxParts = opts.bbox.split(',').map(n => parseInt(n.trim(), 10));
  if (bboxParts.length !== 4) {
    throw new Error('bbox must be 4 comma-separated integers: x,y,w,h');
  }

  if (opts.end === null) {
    throw new Error('--end is required when using --bbox');
  }

  return {
    faces: [{
      character_id: 'face_1',
      bbox: bboxParts,
      start_time_ms: opts.start,
      end_time_ms: opts.end,
    }],
    enhance_quality: opts.enhance,
    fidelity_weight: 0.7,
  };
}

// Main function
async function main() {
  const opts = parseArgs();

  // Validate required options
  if (!opts.video) {
    console.error('Error: --video is required');
    process.exit(1);
  }
  if (!opts.audio) {
    console.error('Error: --audio is required');
    process.exit(1);
  }

  // Check files exist
  if (!existsSync(opts.video)) {
    console.error(`Error: Video file not found: ${opts.video}`);
    process.exit(1);
  }
  if (!existsSync(opts.audio)) {
    console.error(`Error: Audio file not found: ${opts.audio}`);
    process.exit(1);
  }

  // Build request config
  let requestConfig;
  try {
    requestConfig = buildRequestConfig(opts);
  } catch (err) {
    console.error(`Error: ${err.message}`);
    process.exit(1);
  }

  console.log('Lip-Sync API Test');
  console.log('=================');
  console.log(`API:    ${opts.api}`);
  console.log(`Video:  ${opts.video}`);
  console.log(`Audio:  ${opts.audio}`);
  console.log(`Output: ${opts.output}`);
  console.log(`Faces:  ${requestConfig.faces.length}`);
  console.log('');

  // Check API health first
  console.log('Checking API health...');
  try {
    const healthRes = await fetch(`${opts.api}/health`);
    const health = await healthRes.json();
    console.log(`  Status: ${health.status}`);
    console.log(`  Models downloaded: ${health.models_downloaded}`);
    console.log(`  Models loaded: ${health.models_loaded}`);
    console.log(`  GPU: ${health.gpu_available ? `${health.gpu_name} (${health.gpu_memory_gb}GB)` : 'Not available'}`);
    console.log('');
  } catch (err) {
    console.error(`Error: Cannot connect to API at ${opts.api}`);
    console.error(`  ${err.message}`);
    process.exit(1);
  }

  // Read files
  console.log('Reading input files...');
  const videoBuffer = readFileSync(opts.video);
  const audioBuffer = readFileSync(opts.audio);

  // Build form data
  const formData = new FormData();
  formData.append('video', new Blob([videoBuffer]), basename(opts.video));
  formData.append('audio', new Blob([audioBuffer]), basename(opts.audio));
  formData.append('request', JSON.stringify(requestConfig));

  // Make request
  console.log('Sending request to API...');
  console.log(`  Faces: ${JSON.stringify(requestConfig.faces.map(f => f.character_id))}`);
  console.log('');

  const startTime = Date.now();

  try {
    const response = await fetch(`${opts.api}/lipsync`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API returned ${response.status}: ${errorText}`);
    }

    // Get response headers
    const facesProcessed = response.headers.get('X-Faces-Processed');
    const processingTime = response.headers.get('X-Processing-Time-Ms');

    // Save video
    const videoData = await response.arrayBuffer();
    writeFileSync(opts.output, Buffer.from(videoData));

    const totalTime = Date.now() - startTime;

    console.log('Success!');
    console.log('========');
    console.log(`  Output: ${opts.output}`);
    console.log(`  Size: ${(videoData.byteLength / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  Faces processed: ${facesProcessed || 'N/A'}`);
    console.log(`  Server time: ${processingTime ? `${(parseInt(processingTime) / 1000).toFixed(1)}s` : 'N/A'}`);
    console.log(`  Total time: ${(totalTime / 1000).toFixed(1)}s`);

  } catch (err) {
    console.error('Error:', err.message);
    process.exit(1);
  }
}

main();

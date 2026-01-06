#!/usr/bin/env node
/**
 * Test script for the lip-sync API.
 *
 * Usage:
 *   node test-api.mjs --video https://example.com/video.mp4 --audio https://example.com/audio.mp3 --auto
 *   node test-api.mjs --video input.mp4 --audio speech.wav --auto
 *   node test-api.mjs --video input.mp4 --audio speech.wav --bbox 100,50,200,250 --end 5000
 *   node test-api.mjs --video input.mp4 --audio speech.wav --config faces.json
 *
 * Options:
 *   --video      Input video file or URL (required)
 *   --audio      Audio file or URL for lip-sync (required)
 *   --output     Output video file (default: output.mp4)
 *   --auto       Auto-detect faces using Qwen-VL (requires OPENROUTER_API_KEY on server)
 *   --bbox       Face bounding box as x,y,w,h (for single face, alternative to --auto)
 *   --config     JSON file with face configuration (for multi-face)
 *   --start      Start time in ms (default: 0)
 *   --end        End time in ms (default: video duration or 10000 for --auto)
 *   --api        API base URL (default: http://localhost:8000)
 *   --no-enhance Disable CodeFormer enhancement
 *
 * When both --video and --audio are URLs, uses the JSON API (server downloads files).
 * When using local files, uses the multipart API (client uploads files).
 */

import { readFileSync, writeFileSync, existsSync, mkdtempSync, rmSync } from 'fs';
import { basename, join } from 'path';
import { tmpdir } from 'os';
import { execSync } from 'child_process';

// Check if a string is a URL
function isUrl(str) {
  return str.startsWith('http://') || str.startsWith('https://');
}

// Download a file from URL
async function downloadFile(url, destPath) {
  console.log(`  Downloading: ${url}`);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.status}`);
  }
  const buffer = Buffer.from(await response.arrayBuffer());
  writeFileSync(destPath, buffer);
  console.log(`  Downloaded: ${(buffer.length / 1024 / 1024).toFixed(2)} MB`);
  return destPath;
}

// Get filename from URL
function getFilenameFromUrl(url) {
  const urlObj = new URL(url);
  const pathParts = urlObj.pathname.split('/');
  return pathParts[pathParts.length - 1] || 'file';
}

// Extract a frame from video using ffmpeg
function extractFrame(videoPath, outputPath, timeSeconds = 0) {
  console.log(`  Extracting frame at ${timeSeconds}s...`);
  try {
    execSync(
      `ffmpeg -y -ss ${timeSeconds} -i "${videoPath}" -vframes 1 -f image2 "${outputPath}"`,
      { stdio: 'pipe' }
    );
    return true;
  } catch (err) {
    console.error('  Failed to extract frame with ffmpeg');
    return false;
  }
}

// Detect faces using the API (multipart - for local files)
async function detectFaces(apiBase, framePath) {
  console.log('Detecting faces with Qwen-VL (multipart)...');

  const frameBuffer = readFileSync(framePath);
  const formData = new FormData();
  formData.append('frame', new Blob([frameBuffer]), 'frame.jpg');

  // Use empty characters list to detect all faces
  formData.append('request', JSON.stringify({
    characters: [
      { id: 'person', name: 'Person', description: 'Any person visible in the frame' }
    ]
  }));

  const response = await fetch(`${apiBase}/detect-faces`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Face detection failed: ${response.status} - ${errorText}`);
  }

  const result = await response.json();
  console.log(`  Frame size: ${result.frame_width}x${result.frame_height}`);
  console.log(`  Detected ${result.faces.length} face(s)`);

  for (const face of result.faces) {
    console.log(`    - ${face.character_id}: bbox=${JSON.stringify(face.bbox)}, confidence=${face.confidence.toFixed(2)}`);
  }

  return result;
}

// Detect faces using JSON API (for URLs - server downloads video)
// Now uses multi-frame detection with FPS-based sampling
async function detectFacesUrl(apiBase, videoUrl, sampleFps = 3.0) {
  console.log('Detecting faces with Qwen-VL (JSON API - multi-frame)...');
  console.log(`  Video URL: ${videoUrl}`);
  console.log(`  Sample FPS: ${sampleFps}`);

  const response = await fetch(`${apiBase}/detect-faces/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      video_url: videoUrl,
      sample_fps: sampleFps,
      characters: [
        { id: 'person', name: 'Person', description: 'Any person visible in the frame' }
      ]
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Face detection failed: ${response.status} - ${errorText}`);
  }

  const result = await response.json();
  console.log(`  Frame size: ${result.frame_width}x${result.frame_height}`);
  console.log(`  Video duration: ${result.video_duration_ms}ms`);
  console.log(`  Sampled ${result.frames.length} frame(s)`);

  // Show detections per frame
  let totalFaces = 0;
  for (const frame of result.frames) {
    if (frame.faces.length > 0) {
      console.log(`    Frame @${frame.timestamp_ms}ms: ${frame.faces.length} face(s)`);
      for (const face of frame.faces) {
        console.log(`      - ${face.character_id}: bbox=${JSON.stringify(face.bbox)}, confidence=${face.confidence.toFixed(2)}`);
      }
      totalFaces += frame.faces.length;
    }
  }
  console.log(`  Total detections: ${totalFaces}`);

  // Convert multi-frame response to flat faces list (using first frame with detections)
  // This maintains backward compatibility with the rest of the script
  const firstFrameWithFaces = result.frames.find(f => f.faces.length > 0);
  const flatResult = {
    faces: firstFrameWithFaces ? firstFrameWithFaces.faces : [],
    frame_width: result.frame_width,
    frame_height: result.frame_height,
    // Include all frames for advanced usage
    frames: result.frames,
  };

  return flatResult;
}

// Process lip-sync using JSON API (for URLs - server downloads files)
async function lipsyncUrl(apiBase, videoUrl, audioUrl, requestConfig) {
  console.log('Processing lip-sync (JSON API)...');

  const response = await fetch(`${apiBase}/lipsync/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      video_url: videoUrl,
      audio_url: audioUrl,
      ...requestConfig,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API returned ${response.status}: ${errorText}`);
  }

  const result = await response.json();

  if (!result.success) {
    throw new Error(`Lip-sync failed: ${result.error_message}`);
  }

  return result;
}

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    video: null,
    audio: null,
    output: 'output.mp4',
    auto: false,
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
      case '--auto':
        opts.auto = true;
        break;
      case '--no-enhance':
        opts.enhance = false;
        break;
      case '--help':
      case '-h':
        console.log(`
Lip-Sync API Test Script

Usage:
  node test-api.mjs --video input.mp4 --audio speech.wav --auto
  node test-api.mjs --video https://example.com/video.mp4 --audio https://example.com/audio.mp3 --auto
  node test-api.mjs --video input.mp4 --audio speech.wav --bbox 100,50,200,250 --end 5000
  node test-api.mjs --video input.mp4 --audio speech.wav --config faces.json

Options:
  --video      Input video file or URL (required)
  --audio      Audio file or URL for lip-sync (required)
  --output     Output video file (default: output.mp4)
  --auto       Auto-detect faces using Qwen-VL (recommended!)
  --bbox       Face bounding box as x,y,w,h (manual, alternative to --auto)
  --config     JSON file with face configuration (for multi-face)
  --start      Start time in ms (default: 0)
  --end        End time in ms (default: 10000 for --auto)
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
function buildRequestConfig(opts, detectedFaces = null) {
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

  // If auto-detected faces provided
  if (detectedFaces && detectedFaces.length > 0) {
    const endTime = opts.end ?? 10000; // Default 10 seconds for auto mode
    return {
      faces: detectedFaces.map((face, i) => ({
        character_id: face.character_id || `face_${i + 1}`,
        bbox: face.bbox,
        start_time_ms: opts.start,
        end_time_ms: endTime,
      })),
      enhance_quality: opts.enhance,
      fidelity_weight: 0.7,
    };
  }

  // Otherwise build from bbox
  if (!opts.bbox) {
    throw new Error('Either --auto, --bbox, or --config is required');
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

  // Determine API mode: JSON (URLs) or multipart (local files)
  const useJsonApi = isUrl(opts.video) && isUrl(opts.audio);

  console.log('Lip-Sync API Test');
  console.log('=================');
  console.log(`API:    ${opts.api}`);
  console.log(`Video:  ${opts.video}`);
  console.log(`Audio:  ${opts.audio}`);
  console.log(`Output: ${opts.output}`);
  console.log(`Mode:   ${opts.auto ? 'Auto-detect' : opts.config ? 'Config file' : 'Manual bbox'}`);
  console.log(`API:    ${useJsonApi ? 'JSON (server downloads)' : 'Multipart (client uploads)'}`);
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

  // JSON API path (both video and audio are URLs)
  if (useJsonApi) {
    try {
      await runJsonApi(opts);
    } catch (err) {
      console.error('Error:', err.message);
      process.exit(1);
    }
    return;
  }

  // Multipart API path (local files or mixed)
  await runMultipartApi(opts);
}

// Run using JSON API (URLs - server downloads)
async function runJsonApi(opts) {
  const startTime = Date.now();

  // Auto-detect faces if requested
  let detectedFaces = null;
  if (opts.auto) {
    try {
      // Use 3 FPS sampling for face detection
      const detection = await detectFacesUrl(opts.api, opts.video, 3.0);
      detectedFaces = detection.faces;

      if (detectedFaces.length === 0) {
        throw new Error('No faces detected in the video');
      }
    } catch (err) {
      throw new Error(`Face detection failed: ${err.message}`);
    }
    console.log('');
  }

  // Build request config
  let requestConfig;
  try {
    requestConfig = buildRequestConfig(opts, detectedFaces);
  } catch (err) {
    throw err;
  }

  console.log(`Faces:  ${requestConfig.faces.length}`);
  for (const face of requestConfig.faces) {
    console.log(`  - ${face.character_id}: bbox=${JSON.stringify(face.bbox)}, ${face.start_time_ms}ms-${face.end_time_ms}ms`);
  }
  console.log('');

  // Make request
  console.log('Sending request to API...');
  console.log(`  Faces: ${JSON.stringify(requestConfig.faces.map(f => f.character_id))}`);
  console.log('');

  const result = await lipsyncUrl(opts.api, opts.video, opts.audio, requestConfig);

  const totalTime = Date.now() - startTime;

  // Decode base64 video and save
  if (!result.output_url) {
    throw new Error('No output video in response');
  }

  const base64Data = result.output_url.replace('data:video/mp4;base64,', '');
  const videoBuffer = Buffer.from(base64Data, 'base64');
  writeFileSync(opts.output, videoBuffer);

  console.log('Success!');
  console.log('========');
  console.log(`  Output: ${opts.output}`);
  console.log(`  Size: ${(videoBuffer.length / 1024 / 1024).toFixed(2)} MB`);
  console.log(`  Faces processed: ${result.faces_processed}`);
  console.log(`  Server time: ${(result.processing_time_ms / 1000).toFixed(1)}s`);
  console.log(`  Total time: ${(totalTime / 1000).toFixed(1)}s`);
}

// Run using multipart API (local files)
async function runMultipartApi(opts) {
  // Create temp directory for downloads
  let tempDir = null;
  let videoPath = opts.video;
  let audioPath = opts.audio;
  let videoFilename = basename(opts.video);
  let audioFilename = basename(opts.audio);

  try {
    // Download files if URLs provided
    if (isUrl(opts.video) || isUrl(opts.audio)) {
      tempDir = mkdtempSync(join(tmpdir(), 'lipsync-test-'));
      console.log('Downloading files...');

      if (isUrl(opts.video)) {
        videoFilename = getFilenameFromUrl(opts.video);
        videoPath = join(tempDir, videoFilename);
        await downloadFile(opts.video, videoPath);
      }

      if (isUrl(opts.audio)) {
        audioFilename = getFilenameFromUrl(opts.audio);
        audioPath = join(tempDir, audioFilename);
        await downloadFile(opts.audio, audioPath);
      }
      console.log('');
    }

    // Check files exist
    if (!existsSync(videoPath)) {
      console.error(`Error: Video file not found: ${videoPath}`);
      process.exit(1);
    }
    if (!existsSync(audioPath)) {
      console.error(`Error: Audio file not found: ${audioPath}`);
      process.exit(1);
    }

    // Auto-detect faces if requested
    let detectedFaces = null;
    if (opts.auto) {
      // Extract a frame from the video
      const framePath = join(tempDir || mkdtempSync(join(tmpdir(), 'lipsync-frame-')), 'frame.jpg');
      if (!tempDir) {
        tempDir = framePath.replace('/frame.jpg', '');
      }

      if (!extractFrame(videoPath, framePath, 0.5)) {
        console.error('Error: Could not extract frame from video (is ffmpeg installed?)');
        process.exit(1);
      }

      // Detect faces
      try {
        const detection = await detectFaces(opts.api, framePath);
        detectedFaces = detection.faces;

        if (detectedFaces.length === 0) {
          console.error('Error: No faces detected in the video');
          process.exit(1);
        }
      } catch (err) {
        console.error(`Error detecting faces: ${err.message}`);
        process.exit(1);
      }
      console.log('');
    }

    // Build request config
    let requestConfig;
    try {
      requestConfig = buildRequestConfig(opts, detectedFaces);
    } catch (err) {
      console.error(`Error: ${err.message}`);
      process.exit(1);
    }

    console.log(`Faces:  ${requestConfig.faces.length}`);
    for (const face of requestConfig.faces) {
      console.log(`  - ${face.character_id}: bbox=${JSON.stringify(face.bbox)}, ${face.start_time_ms}ms-${face.end_time_ms}ms`);
    }
    console.log('');

    // Read files
    console.log('Reading input files...');
    const videoBuffer = readFileSync(videoPath);
    const audioBuffer = readFileSync(audioPath);

    // Build form data
    const formData = new FormData();
    formData.append('video', new Blob([videoBuffer]), videoFilename);
    formData.append('audio', new Blob([audioBuffer]), audioFilename);
    formData.append('request', JSON.stringify(requestConfig));

    // Make request
    console.log('Sending request to API...');
    console.log(`  Faces: ${JSON.stringify(requestConfig.faces.map(f => f.character_id))}`);
    console.log('');

    const startTime = Date.now();

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
  } finally {
    // Cleanup temp directory
    if (tempDir) {
      try {
        rmSync(tempDir, { recursive: true });
      } catch (e) {
        // Ignore cleanup errors
      }
    }
  }
}

main();

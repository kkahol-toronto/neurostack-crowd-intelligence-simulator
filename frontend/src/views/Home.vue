<template>
  <div class="home-container">
    <div class="bg-mesh" aria-hidden="true" />
    <!-- Top navigation -->
    <nav class="navbar glass-nav">
      <div class="nav-brand">NEUROSTACK_CROWD_INTELLIGENCE_SIMULATOR</div>
      <div class="nav-links">
        <a href="https://github.com/kkahol-toronto/neurostack-crowd-intelligence-simulator" target="_blank" class="github-link">
          Visit our GitHub <span class="arrow">↗</span>
        </a>
      </div>
    </nav>

    <div class="main-content">
      <!-- Hero -->
      <section class="hero-section">
        <div class="hero-left glass-panel glass-panel--hero">
          <div class="tag-row">
            <span class="orange-tag">Simple, universal swarm intelligence</span>
            <span class="version-text">/ v0.1-preview</span>
          </div>
          
          <h1 class="main-title">
            Upload any report<br>
            <span class="gradient-text">Simulate what comes next</span>
          </h1>
          
          <div class="hero-desc">
            <p>
              From a single paragraph of seed text, <span class="highlight-bold">NEUROSTACK_CROWD_INTELLIGENCE_SIMULATOR</span> can spin up a parallel world of up to <span class="highlight-orange">millions of agents</span>. Inject variables from a god’s-eye view and search for <span class="highlight-code">“local optima”</span> in complex group dynamics.
            </p>
            <p class="slogan-text">
              Rehearse the future in agent swarms; win decisions after many simulated battles<span class="blinking-cursor">_</span>
            </p>
          </div>
           
          <div class="decoration-square"></div>
        </div>
        
        <div class="hero-right glass-panel glass-panel--media">
          <!-- Logo -->
          <div class="logo-container">
            <img src="../assets/logo/neurostack_logo_left.jpeg" alt="NEUROSTACK_CROWD_INTELLIGENCE_SIMULATOR logo" class="hero-logo" />
          </div>
          
          <button class="scroll-down-btn" @click="scrollToBottom">
            ↓
          </button>
        </div>
      </section>

      <!-- Two-column dashboard -->
      <section class="dashboard-section">
        <!-- Left: status & steps -->
        <div class="left-panel glass-panel glass-panel--tall">
          <div class="panel-header">
            <span class="status-dot">■</span> System status
          </div>
          
          <h2 class="section-title">Ready</h2>
          <p class="section-desc">
            Prediction engine idle — upload unstructured data to start a simulation run
          </p>
          
          <!-- Metric cards -->
          <div class="metrics-row">
            <div class="metric-card">
              <div class="metric-value">Low cost</div>
              <div class="metric-label">~$5 per typical run</div>
            </div>
            <div class="metric-card">
              <div class="metric-value">Scale</div>
              <div class="metric-label">Up to millions of agents</div>
            </div>
          </div>

          <!-- Workflow steps -->
          <div class="steps-container">
            <div class="steps-header">
               <span class="diamond-icon">◇</span> Workflow
            </div>
            <div class="workflow-list">
              <div class="workflow-item">
                <span class="step-num">01</span>
                <div class="step-info">
                  <div class="step-title">Graph build</div>
                  <div class="step-desc">Seed extraction · individual & group memory · GraphRAG</div>
                </div>
              </div>
              <div class="workflow-item">
                <span class="step-num">02</span>
                <div class="step-info">
                  <div class="step-title">Environment setup</div>
                  <div class="step-desc">Entity/relation extraction · personas · simulation parameters</div>
                </div>
              </div>
              <div class="workflow-item">
                <span class="step-num">03</span>
                <div class="step-info">
                  <div class="step-title">Run simulation</div>
                  <div class="step-desc">Dual-platform run · prediction parsing · temporal memory updates</div>
                </div>
              </div>
              <div class="workflow-item">
                <span class="step-num">04</span>
                <div class="step-info">
                  <div class="step-title">Report</div>
                  <div class="step-desc">ReportAgent tools · deep interaction with the post-run world</div>
                </div>
              </div>
              <div class="workflow-item">
                <span class="step-num">05</span>
                <div class="step-info">
                  <div class="step-title">Deep interaction</div>
                  <div class="step-desc">Talk to any agent · chat with ReportAgent</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Right: console -->
        <div class="right-panel">
          <div class="console-box glass-panel glass-panel--console">
            <!-- Upload -->
            <div class="console-section">
              <div class="console-header">
                <span class="console-label">01 / Seed data</span>
                <span class="console-meta">Formats: PDF, MD, TXT</span>
              </div>
              
              <div 
                class="upload-zone"
                :class="{ 'drag-over': isDragOver, 'has-files': files.length > 0 }"
                @dragover.prevent="handleDragOver"
                @dragleave.prevent="handleDragLeave"
                @drop.prevent="handleDrop"
                @click="triggerFileInput"
              >
                <input
                  ref="fileInput"
                  type="file"
                  multiple
                  accept=".pdf,.md,.txt"
                  @change="handleFileSelect"
                  style="display: none"
                  :disabled="loading"
                />
                
                <div v-if="files.length === 0" class="upload-placeholder">
                  <div class="upload-icon">↑</div>
                  <div class="upload-title">Drop files here</div>
                  <div class="upload-hint">or click to browse</div>
                </div>
                
                <div v-else class="file-list">
                  <div v-for="(file, index) in files" :key="index" class="file-item">
                    <span class="file-icon">📄</span>
                    <span class="file-name">{{ file.name }}</span>
                    <button @click.stop="removeFile(index)" class="remove-btn">×</button>
                  </div>
                </div>
              </div>
            </div>

            <!-- Divider -->
            <div class="console-divider">
              <span>Parameters</span>
            </div>

            <!-- Prompt input -->
            <div class="console-section">
              <div class="console-header">
                <span class="console-label">>_ 02 / Simulation prompt</span>
              </div>
              <div class="input-wrapper">
                <textarea
                  v-model="formData.simulationRequirement"
                  class="code-input"
                  placeholder="// Describe what to simulate or predict in natural language (e.g. policy change X — how might public opinion evolve?)"
                  rows="6"
                  :disabled="loading"
                ></textarea>
                <div class="model-badge">Engine: NEUROSTACK_CIS v1.0</div>
              </div>
            </div>

            <!-- Start -->
            <div class="console-section btn-section">
              <button 
                class="start-engine-btn"
                @click="startSimulation"
                :disabled="!canSubmit || loading"
              >
                <span v-if="!loading">Start engine</span>
                <span v-else>Initializing...</span>
                <span class="btn-arrow">→</span>
              </button>
            </div>
          </div>
        </div>
      </section>

      <!-- History -->
      <div class="history-wrap">
        <HistoryDatabase />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import HistoryDatabase from '../components/HistoryDatabase.vue'

const router = useRouter()

// Form state
const formData = ref({
  simulationRequirement: ''
})

// Uploaded files
const files = ref([])

// UI state
const loading = ref(false)
const error = ref('')
const isDragOver = ref(false)

const fileInput = ref(null)

// Submit enabled when prompt + at least one file
const canSubmit = computed(() => {
  return formData.value.simulationRequirement.trim() !== '' && files.value.length > 0
})

const triggerFileInput = () => {
  if (!loading.value) {
    fileInput.value?.click()
  }
}

const handleFileSelect = (event) => {
  const selectedFiles = Array.from(event.target.files)
  addFiles(selectedFiles)
}

const handleDragOver = (e) => {
  if (!loading.value) {
    isDragOver.value = true
  }
}

const handleDragLeave = (e) => {
  isDragOver.value = false
}

const handleDrop = (e) => {
  isDragOver.value = false
  if (loading.value) return
  
  const droppedFiles = Array.from(e.dataTransfer.files)
  addFiles(droppedFiles)
}

const addFiles = (newFiles) => {
  const validFiles = newFiles.filter(file => {
    const ext = file.name.split('.').pop().toLowerCase()
    return ['pdf', 'md', 'txt'].includes(ext)
  })
  files.value.push(...validFiles)
}

const removeFile = (index) => {
  files.value.splice(index, 1)
}

const scrollToBottom = () => {
  window.scrollTo({
    top: document.body.scrollHeight,
    behavior: 'smooth'
  })
}

// Navigate to Process; upload API runs there
const startSimulation = () => {
  if (!canSubmit.value || loading.value) return
  
  import('../store/pendingUpload.js').then(({ setPendingUpload }) => {
    setPendingUpload(files.value, formData.value.simulationRequirement)
    
    // `new` project id — Process handles upload
    router.push({
      name: 'Process',
      params: { projectId: 'new' }
    })
  })
}
</script>

<style scoped>
/* Neon + glass theme (scoped to home) */
.home-container {
  --void: #0a0a0f;
  --void-deep: #050508;
  --neon-orange: #ff6b35;
  --neon-green: #39ff9e;
  --neon-blue: #00d4ff;
  --neon-mint: #7dffcb;
  --text-muted: rgba(200, 230, 255, 0.7);
  --glass: rgba(255, 255, 255, 0.06);
  --glass-border: rgba(255, 255, 255, 0.12);
  --glass-highlight: rgba(255, 255, 255, 0.1);
  --font-mono: 'JetBrains Mono', monospace;
  --font-sans: 'Space Grotesk', system-ui, sans-serif;

  position: relative;
  min-height: 100vh;
  background: var(--void-deep);
  font-family: var(--font-sans);
  color: rgba(240, 245, 255, 0.92);
  overflow-x: hidden;
}

.bg-mesh {
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background:
    radial-gradient(ellipse 80% 50% at 20% 0%, rgba(255, 107, 53, 0.15) 0%, transparent 55%),
    radial-gradient(ellipse 80% 50% at 80% 100%, rgba(0, 212, 255, 0.12) 0%, transparent 50%),
    radial-gradient(ellipse 60% 40% at 50% 50%, rgba(57, 255, 158, 0.06) 0%, transparent 45%);
  animation: mesh-shift 18s ease-in-out infinite alternate;
}

@keyframes mesh-shift {
  0% { opacity: 1; }
  100% { opacity: 0.85; filter: hue-rotate(15deg); }
}

.home-container > :not(.bg-mesh) {
  position: relative;
  z-index: 1;
}

/* Glass surfaces */
.glass-panel {
  backdrop-filter: blur(18px) saturate(160%);
  -webkit-backdrop-filter: blur(18px) saturate(160%);
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  box-shadow:
    0 8px 32px rgba(0, 0, 0, 0.45),
    inset 0 1px 0 var(--glass-highlight);
  transition:
    transform 0.4s cubic-bezier(0.22, 1, 0.36, 1),
    box-shadow 0.35s ease,
    border-color 0.3s ease;
}

.glass-panel:hover {
  border-color: rgba(0, 212, 255, 0.28);
  box-shadow:
    0 16px 48px rgba(0, 212, 255, 0.07),
    0 8px 32px rgba(0, 0, 0, 0.5),
    inset 0 1px 0 rgba(255, 255, 255, 0.14);
}

.glass-panel--hero {
  padding: 28px 32px 32px;
  margin-right: 8px;
}

.glass-panel--media {
  padding: 24px;
  align-items: center;
}

.glass-panel--tall {
  padding: 28px;
}

.glass-panel--console {
  padding: 4px;
  border-radius: 14px;
}

/* Navbar */
.navbar {
  height: 64px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 clamp(20px, 4vw, 48px);
}

.glass-nav {
  backdrop-filter: blur(22px) saturate(180%);
  -webkit-backdrop-filter: blur(22px) saturate(180%);
  background: rgba(8, 8, 12, 0.72) !important;
  border-bottom: 1px solid var(--glass-border);
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
}

.nav-brand {
  font-family: var(--font-mono);
  font-weight: 800;
  letter-spacing: 0.06em;
  font-size: clamp(0.45rem, 0.65vw + 0.35rem, 0.72rem);
  line-height: 1.2;
  max-width: min(96vw, 52rem);
  color: rgba(240, 248, 255, 0.95);
  text-shadow: 0 0 24px rgba(0, 212, 255, 0.25);
}

.nav-links {
  display: flex;
  align-items: center;
}

.github-link {
  color: var(--neon-green);
  text-decoration: none;
  font-family: var(--font-mono);
  font-size: 0.88rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  border-radius: 999px;
  border: 1px solid rgba(57, 255, 158, 0.25);
  background: rgba(57, 255, 158, 0.06);
  transition: color 0.25s ease, box-shadow 0.3s ease, transform 0.2s ease, border-color 0.25s ease;
}

.github-link:hover {
  color: var(--neon-blue);
  border-color: rgba(0, 212, 255, 0.45);
  box-shadow: 0 0 24px rgba(0, 212, 255, 0.25);
  transform: translateY(-1px);
}

.arrow {
  font-family: sans-serif;
}

/* Main */
.main-content {
  max-width: 1400px;
  margin: 0 auto;
  padding: clamp(32px, 5vw, 60px) clamp(20px, 4vw, 40px);
}

/* Hero */
.hero-section {
  display: flex;
  justify-content: space-between;
  align-items: stretch;
  gap: 28px;
  margin-bottom: clamp(48px, 8vw, 80px);
  position: relative;
}

.hero-left {
  flex: 1;
  min-width: 0;
  padding-right: 0;
}

.tag-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.75rem 15px;
  margin-bottom: 25px;
  font-family: var(--font-mono);
  font-size: 0.8rem;
}

.orange-tag {
  background: linear-gradient(135deg, rgba(255, 107, 53, 0.35), rgba(255, 107, 53, 0.15));
  color: var(--neon-orange);
  padding: 6px 12px;
  font-weight: 700;
  letter-spacing: 1px;
  font-size: 0.75rem;
  border-radius: 6px;
  border: 1px solid rgba(255, 107, 53, 0.45);
  box-shadow: 0 0 20px rgba(255, 107, 53, 0.2);
}

.version-text {
  color: var(--neon-blue);
  font-weight: 500;
  letter-spacing: 0.5px;
  text-shadow: 0 0 12px rgba(0, 212, 255, 0.35);
}

.main-title {
  font-size: clamp(2.2rem, 6vw, 4.5rem);
  line-height: 1.15;
  font-weight: 600;
  margin: 0 0 40px 0;
  letter-spacing: -0.02em;
  color: rgba(240, 245, 255, 0.92);
  text-shadow: 0 0 40px rgba(0, 212, 255, 0.12);
}

.gradient-text {
  background: linear-gradient(90deg, var(--neon-orange) 0%, #ff9f7a 40%, var(--neon-green) 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  display: inline-block;
  filter: drop-shadow(0 0 20px rgba(255, 107, 53, 0.35));
}

.hero-desc {
  font-size: 1.05rem;
  line-height: 1.8;
  color: var(--text-muted);
  max-width: 640px;
  margin-bottom: 50px;
  font-weight: 400;
  text-align: justify;
}

.hero-desc p {
  margin-bottom: 1.5rem;
}

.highlight-bold {
  color: var(--neon-blue);
  font-weight: 700;
  text-shadow: 0 0 12px rgba(0, 212, 255, 0.35);
}

.highlight-orange {
  color: var(--neon-orange);
  font-weight: 700;
  font-family: var(--font-mono);
  text-shadow: 0 0 14px rgba(255, 107, 53, 0.35);
}

.highlight-code {
  background: rgba(0, 212, 255, 0.1);
  padding: 2px 8px;
  border-radius: 6px;
  font-family: var(--font-mono);
  font-size: 0.9em;
  color: var(--neon-green);
  font-weight: 600;
  border: 1px solid rgba(57, 255, 158, 0.25);
}

.slogan-text {
  font-size: 1.2rem;
  font-weight: 520;
  color: var(--neon-mint);
  letter-spacing: 0.04em;
  border-left: 3px solid var(--neon-blue);
  padding-left: 15px;
  margin-top: 20px;
  text-shadow: 0 0 16px rgba(57, 255, 158, 0.2);
}

.blinking-cursor {
  color: var(--neon-orange);
  animation: blink 1s step-end infinite;
  font-weight: 700;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

.decoration-square {
  width: 16px;
  height: 16px;
  background: var(--neon-orange);
  box-shadow: 0 0 16px rgba(255, 107, 53, 0.7);
  border-radius: 2px;
}

.hero-right {
  flex: 0.85;
  min-width: 0;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-items: stretch;
}

.logo-container {
  width: 100%;
  display: flex;
  justify-content: center;
  padding: 8px 14px 0;
}

.hero-logo {
  max-width: min(500px, 100%);
  width: 100%;
  border-radius: 8px;
  transition: transform 0.5s cubic-bezier(0.22, 1, 0.36, 1);
}

.glass-panel--media:hover .hero-logo {
  transform: scale(1.02);
}

.scroll-down-btn {
  width: 44px;
  height: 44px;
  align-self: flex-end;
  margin-top: 1rem;
  border: 1px solid rgba(0, 212, 255, 0.35);
  border-radius: 10px;
  background: rgba(0, 212, 255, 0.08);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: var(--neon-blue);
  font-size: 1.2rem;
  transition: all 0.3s ease;
  box-shadow: 0 0 20px rgba(0, 212, 255, 0.15);
}

.scroll-down-btn:hover {
  border-color: var(--neon-orange);
  color: var(--neon-orange);
  box-shadow: 0 0 28px rgba(255, 107, 53, 0.35);
  transform: translateY(0.25rem);
}

/* Dashboard */
.dashboard-section {
  display: flex;
  gap: clamp(24px, 4vw, 48px);
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  padding-top: clamp(40px, 6vw, 60px);
  align-items: stretch;
}

.dashboard-section .left-panel,
.dashboard-section .right-panel {
  display: flex;
  flex-direction: column;
}

/* Left column */
.left-panel {
  flex: 0.85;
  min-width: 0;
}

.panel-header {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  color: var(--neon-blue);
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 20px;
  text-shadow: 0 0 10px rgba(0, 212, 255, 0.25);
}

.status-dot {
  color: var(--neon-orange);
  font-size: 0.8rem;
  animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
}

.section-title {
  font-size: clamp(1.5rem, 3vw, 2rem);
  font-weight: 600;
  margin: 0 0 15px 0;
  color: var(--neon-green);
  text-shadow: 0 0 20px rgba(57, 255, 158, 0.25);
}

.section-desc {
  color: var(--text-muted);
  margin-bottom: 25px;
  line-height: 1.6;
}

.metrics-row {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 15px;
}

.metric-card {
  border: 1px solid rgba(0, 212, 255, 0.2);
  padding: 18px 22px;
  min-width: 140px;
  border-radius: 12px;
  background: rgba(0, 212, 255, 0.05);
  transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}

.metric-card:hover {
  transform: translateY(-3px);
  border-color: rgba(255, 107, 53, 0.35);
  box-shadow: 0 8px 24px rgba(255, 107, 53, 0.12);
}

.metric-value {
  font-family: var(--font-mono);
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 5px;
  color: var(--neon-orange);
}

.metric-label {
  font-size: 0.85rem;
  color: var(--text-muted);
}

/* Workflow steps */
.steps-container {
  border: 1px solid rgba(255, 255, 255, 0.1);
  padding: 24px;
  position: relative;
  border-radius: 12px;
  background: rgba(0, 0, 0, 0.2);
}

.steps-header {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  color: var(--neon-blue);
  margin-bottom: 25px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.diamond-icon {
  font-size: 1.2rem;
  line-height: 1;
  color: var(--neon-green);
}

.workflow-list {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.workflow-item {
  display: flex;
  align-items: flex-start;
  gap: 20px;
  padding: 10px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  transition: background 0.25s ease;
  border-radius: 8px;
  margin: 0 -8px;
  padding-left: 8px;
  padding-right: 8px;
}

.workflow-item:last-child {
  border-bottom: none;
}

.workflow-item:hover {
  background: rgba(0, 212, 255, 0.06);
}

.step-num {
  font-family: var(--font-mono);
  font-weight: 700;
  color: var(--neon-blue);
  opacity: 0.85;
}

.step-info {
  flex: 1;
}

.step-title {
  font-weight: 600;
  font-size: 1rem;
  margin-bottom: 4px;
  color: rgba(245, 250, 255, 0.95);
}

.step-desc {
  font-size: 0.85rem;
  color: var(--text-muted);
}

/* Right column / console */
.right-panel {
  flex: 1.15;
  min-width: 0;
}

.console-box {
  padding: 6px;
}

.console-section {
  padding: 18px 20px;
}

.console-section.btn-section {
  padding-top: 8px;
  padding-bottom: 18px;
}

.console-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 15px;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--neon-blue);
}

.console-meta {
  color: var(--text-muted);
}

.upload-zone {
  border: 1px dashed rgba(0, 212, 255, 0.35);
  border-radius: 12px;
  height: 200px;
  overflow-y: auto;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: border-color 0.3s ease, background 0.3s ease, box-shadow 0.3s ease;
  background: rgba(0, 0, 0, 0.25);
}

.upload-zone.drag-over {
  border-color: var(--neon-orange);
  background: rgba(255, 107, 53, 0.1);
  box-shadow: inset 0 0 40px rgba(255, 107, 53, 0.15);
}

.upload-zone.has-files {
  align-items: flex-start;
}

.upload-zone:hover {
  border-color: rgba(57, 255, 158, 0.45);
  background: rgba(57, 255, 158, 0.06);
}

.upload-placeholder {
  text-align: center;
}

.upload-icon {
  width: 44px;
  height: 44px;
  border: 1px solid rgba(0, 212, 255, 0.35);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 15px;
  color: var(--neon-blue);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.upload-zone:hover .upload-icon {
  transform: translateY(-2px);
  box-shadow: 0 0 20px rgba(0, 212, 255, 0.25);
}

.upload-title {
  font-weight: 600;
  font-size: 0.95rem;
  margin-bottom: 5px;
  color: rgba(245, 250, 255, 0.95);
}

.upload-hint {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--text-muted);
}

.file-list {
  width: 100%;
  padding: 15px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.file-item {
  display: flex;
  align-items: center;
  background: rgba(255, 255, 255, 0.04);
  padding: 10px 12px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 8px;
  font-family: var(--font-mono);
  font-size: 0.85rem;
  color: var(--neon-mint);
  transition: border-color 0.2s ease;
}

.file-item:hover {
  border-color: rgba(0, 212, 255, 0.3);
}

.file-name {
  flex: 1;
  margin: 0 10px;
  color: rgba(240, 245, 255, 0.9);
}

.remove-btn {
  background: rgba(255, 107, 53, 0.15);
  border: none;
  cursor: pointer;
  font-size: 1.2rem;
  color: var(--neon-orange);
  border-radius: 6px;
  padding: 0 8px;
  line-height: 1;
  transition: background 0.2s ease;
}

.remove-btn:hover {
  background: rgba(255, 107, 53, 0.35);
}

.console-divider {
  display: flex;
  align-items: center;
  margin: 10px 0;
}

.console-divider::before,
.console-divider::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.35), transparent);
}

.console-divider span {
  padding: 0 15px;
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--neon-green);
  letter-spacing: 1px;
}

.input-wrapper {
  position: relative;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  background: rgba(0, 0, 0, 0.35);
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}

.input-wrapper:focus-within {
  border-color: rgba(0, 212, 255, 0.45);
  box-shadow: 0 0 0 1px rgba(0, 212, 255, 0.15);
}

.code-input {
  width: 100%;
  border: none;
  background: transparent;
  padding: 20px;
  font-family: var(--font-mono);
  font-size: 0.9rem;
  line-height: 1.6;
  resize: vertical;
  outline: none;
  min-height: 150px;
  color: rgba(230, 245, 255, 0.92);
}

.code-input::placeholder {
  color: rgba(160, 190, 210, 0.45);
}

.model-badge {
  position: absolute;
  bottom: 10px;
  right: 15px;
  font-family: var(--font-mono);
  font-size: clamp(0.45rem, 0.55vw + 0.35rem, 0.65rem);
  color: var(--text-muted);
  max-width: calc(100% - 30px);
  text-align: right;
  line-height: 1.25;
}

.start-engine-btn {
  width: 100%;
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(255, 107, 53, 0.9), rgba(255, 80, 40, 0.85));
  color: #0a0a0f;
  border: 1px solid rgba(255, 200, 160, 0.35);
  padding: 20px;
  font-family: var(--font-mono);
  font-weight: 700;
  font-size: 1.1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  transition: transform 0.25s ease, box-shadow 0.35s ease, filter 0.25s ease;
  letter-spacing: 1px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 4px 24px rgba(255, 107, 53, 0.35);
}

.start-engine-btn::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
  transform: translateX(-100%);
  transition: transform 0.6s ease;
}

.start-engine-btn:hover:not(:disabled)::before {
  transform: translateX(100%);
}

.start-engine-btn:not(:disabled) {
  animation: pulse-glow 2.5s ease-in-out infinite;
}

.start-engine-btn:hover:not(:disabled) {
  transform: translateY(-3px);
  box-shadow: 0 12px 40px rgba(255, 107, 53, 0.45);
  filter: brightness(1.05);
}

.start-engine-btn:active:not(:disabled) {
  transform: translateY(-1px);
}

.start-engine-btn:disabled {
  background: rgba(80, 80, 90, 0.5);
  color: rgba(180, 180, 190, 0.6);
  cursor: not-allowed;
  border-color: rgba(255, 255, 255, 0.08);
  box-shadow: none;
  animation: none;
}

@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 4px 24px rgba(255, 107, 53, 0.35); }
  50% { box-shadow: 0 4px 32px rgba(0, 212, 255, 0.2), 0 4px 24px rgba(255, 107, 53, 0.4); }
}

/* Responsive */
@media (max-width: 1024px) {
  .dashboard-section {
    flex-direction: column;
  }

  .hero-section {
    flex-direction: column;
  }

  .hero-left {
    margin-bottom: 0;
  }

  .glass-panel--hero {
    margin-right: 0;
  }

  .hero-logo {
    max-width: min(280px, 100%);
    margin-bottom: 12px;
  }

  .navbar {
    height: auto;
    min-height: 56px;
    padding: 12px clamp(16px, 3vw, 40px);
    flex-wrap: wrap;
    gap: 12px;
  }
}

@media (max-width: 600px) {
  .metrics-row {
    flex-direction: column;
  }

  .metric-card {
    width: 100%;
  }
}

.history-wrap {
  margin-top: clamp(32px, 6vw, 56px);
  padding-top: clamp(24px, 4vw, 40px);
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}
</style>

<template>
  <router-view />
</template>

<script setup>
// Routed pages via Vue Router
</script>

<style>
/* Global reset */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html,
body {
  min-height: 100%;
  background: #050508;
}

#app {
  font-family: 'JetBrains Mono', 'Space Grotesk', 'Noto Sans SC', monospace;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: rgba(240, 245, 255, 0.92);
  background-color: #050508;
  min-height: 100vh;
}

/* Scrollbars — dark shell */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #12121a;
}

::-webkit-scrollbar-thumb {
  background: rgba(0, 212, 255, 0.35);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 107, 53, 0.45);
}

/* Buttons inherit font */
button {
  font-family: inherit;
}
</style>

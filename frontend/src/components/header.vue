<template>
  <v-card color="blue lighten-5" elevation="2">
    <v-card-title class="text-center justify-center py-6">
      <h1 class="font-weight-bold" style="word-break: normal">
        Depth Assisted Super Resolution
      </h1>
    </v-card-title>

    <v-tabs background-color="transparent" color="black" grow>
      <v-tab v-for="tab in tabs" :key="tab.url" :to="tab.url">
        <h2>{{ tab.name }}</h2>
      </v-tab>
    </v-tabs>
  </v-card>
</template>

<script>
export default {
  props: { tabs: Array },
};
</script>
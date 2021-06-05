<template>
  <v-app>
    <Header :tabs="tabs" />
    <router-view />
  </v-app>
</template>


<script>
import Header from "./components/header.vue";
export default {
  components: {
    Header,
  },
  data: () => ({
    tabs: [
      { name: "Home", url: "/" },
      { name: "Demo", url: "/demo" },
    ],
  }),
};
</script>

<style>
.theme {
  color: #003e74;
}
</style>
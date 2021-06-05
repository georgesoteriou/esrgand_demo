<template>
  <fragment>
    <h1>
      {{ title }}
      <v-btn icon :href="img" :download="name">
        <v-icon>mdi-download</v-icon>
      </v-btn>
    </h1>
    <panZoom v-if="img != null && !disable" class="zoomBorder mx-5">
      <v-img :src="img" contain />
    </panZoom>
    <v-img v-if="img != null && disable" :src="img" contain />
    <v-skeleton-loader v-if="img == null" type="image" min-height="500px" />
  </fragment>
</template>

<script>
export default {
  props: ["img", "title", "disable"],
  computed: {
    zoom() {
      if (this.disable) {
        return "nothing";
      } else {
        return "";
      }
    },
    name() {
      return `${this.title.toLowerCase().split(" ").join("_")}.jpg`;
    },
  },
};
</script>

<style scoped>
.v-skeleton-loader {
  height: 95%;
}

.v-skeleton-loader__image {
  height: 100%;
}

.zoomBorder {
  overflow: hidden;
  background-color: #ccc;
  border-style: solid;
}
</style>
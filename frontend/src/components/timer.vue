<template>
  <div>
    <v-progress-linear
      v-if="time != 0"
      striped
      height="10"
      :value="percentLeft"
    ></v-progress-linear>
  </div>
</template>

<script>
export default {
  props: ["loading"],
  data() {
    return {
      time: 0,
      interval: null,
      totalTime: 60,
    };
  },
  mounted() {
    fetch(`${process.env.VUE_APP_SERVER}/api/timeout`)
      .then((resp) => resp.json())
      .then((data) => {
        this.totalTime = data["timeout"];
      });
  },
  watch: {
    loading(is) {
      if (is) {
        this.interval = setInterval(this.incrementTime, this.totalTime * 10);
      } else {
        clearInterval(this.interval);
        this.time = 100;
        setTimeout(() => (this.time = 0), 500);
      }
    },
  },
  methods: {
    incrementTime() {
      this.time = this.time + this.totalTime / 100;
    },
  },
  computed: {
    percentLeft() {
      return (this.time / this.totalTime) * 100;
    },
  },
};
</script>
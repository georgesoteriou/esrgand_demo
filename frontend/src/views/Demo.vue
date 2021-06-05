<template>
  <v-container class="my-10" fluid>
    <v-row justify="center">
      <v-col cols="12" md="9" xl="6">
        <v-file-input
          label="Upload your image"
          filled
          outlined
          :disabled="loading"
          prepend-icon="mdi-camera"
          accept="image/png, image/jpeg"
          v-model="file"
        ></v-file-input>
      </v-col>
      <v-col cols="auto">
        <v-btn
          xLarge
          class="white--text"
          color="#003E74"
          @click="enhance"
          :loading="loading"
        >
          Enhance!
        </v-btn>
      </v-col>
    </v-row>
    <v-row min-height="10">
      <v-col cols="12"><Timer :loading="loading" /> </v-col>
    </v-row>
    <v-alert dismissible type="error" v-if="error">{{ error }}</v-alert>
    <v-row justify="center">
      <v-col cols="12" lg="6" align="center" v-if="input">
        <ZoomImage title="Your Image" :img="input" :disable="!zoom" />
      </v-col>
      <v-col cols="12" lg="6" align="center" v-if="(input && loading) || depth">
        <ZoomImage title="Depth Estimation" :img="depth" :disable="!zoom" />
      </v-col>
    </v-row>
    <v-row justify="center" min-height="500px">
      <v-col cols="12" lg="6" align="center" v-if="(input && loading) || hrx4">
        <ZoomImage title="Bicubic Enlargement" :img="hrx4" :disable="!zoom" />
      </v-col>
      <v-col cols="12" lg="6" align="center" v-if="(input && loading) || sr">
        <ZoomImage title="ESRGAN_D SR" :img="sr" :disable="!zoom" />
      </v-col>
    </v-row>
    <v-btn
      @click="zoom = !zoom"
      :dark="zoom"
      fab
      fixed
      bottom
      right
      large
      :ripple="false"
      :color="col"
    >
      <v-icon>mdi-magnify-plus-outline</v-icon>
    </v-btn>
  </v-container>
</template>

<script>
import ZoomImage from "../components/zoom-image";
import Timer from "../components/timer";

export default {
  components: { ZoomImage, Timer },
  data() {
    return {
      file: null,
      loading: false,
      sr: null,
      depth: null,
      hrx4: null,
      error: null,
      zoom: false,
      totalTime: 60,
      time: 0,
    };
  },
  computed: {
    input() {
      if (this.file) {
        return URL.createObjectURL(this.file);
      } else {
        return null;
      }
    },
    col() {
      if (this.zoom) {
        return "#003E74";
      } else {
        return "white";
      }
    },
  },
  watch: {
    input(curr, prev) {
      if (curr != prev) {
        this.depth = null;
        this.sr = null;
        this.hrx4 = null;
      }
    },
  },
  methods: {
    enhance() {
      if (this.file) {
        this.error = null;
        this.loading = true;
        let data = new FormData();
        data.append("file", this.file);
        fetch(`${process.env.VUE_APP_SERVER}/api/enhance`, {
          method: "POST",
          body: data,
        })
          .then((resp) => resp.json())
          .then((data) => {
            if (data["success"]) {
              this.sr = data["sr"];
              this.depth = data["depth"];
              this.hrx4 = data["hrx4"];
            } else {
              this.error = data["msg"];
            }
            this.loading = false;
          })
          .catch(() => {
            // this.loading = false;
          });
      }
    },
  },
};
</script>
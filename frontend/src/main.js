import Vue from 'vue'
import App from './App.vue'
import './registerServiceWorker'
import vuetify from './plugins/vuetify'
import router from './router'
import Fragment from 'vue-fragment'
import panZoom from 'vue-panzoom'

Vue.config.productionTip = false

new Vue({
  vuetify,
  router,
  render: h => h(App)
}).$mount('#app')

Vue.use(Fragment.Plugin)
Vue.use(panZoom);


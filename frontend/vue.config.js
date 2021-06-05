module.exports = {
  chainWebpack: config => {
      config
          .plugin('html')
          .tap(args => {
              args[0].title = "Super Resolution ESRGAN_D";
              return args;
          })
  },
  transpileDependencies: [
    'vuetify'
    
  ]
}

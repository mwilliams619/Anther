const path = require('path');
const webpack = require('webpack');

module.exports = {
  entry: './static/js/fin_dash.js', // Entry point of your application
  mode: 'development', 
  devServer: {
    hot: true,
  },
  plugins: [
    new webpack.HotModuleReplacementPlugin()
  ],
  output: {
    filename: 'bundle2.js', // Output file name
    path: path.resolve(__dirname, './static/js') // Output directory
  },
  module: {
    rules: [
      {
        test: /\.jsx?$/, // Transpile JavaScript and JSX files
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env', '@babel/preset-react'],
            targets: "IE 11, > 0.5%, last 2 versions" // ES5
          }
        },
        resolve: {
            extensions: ['.js', '.jsx'],
          },
    },
    {
      test: /\.css$/,
      use: ['style-loader', 'css-loader'] 
    }
    ]
  }
};

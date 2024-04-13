import React from 'react';
import ReactDOM from 'react-dom/client';
import HelloWorld, {MetricsTracker} from '../react/spend_metrics.jsx';
import { Add } from '../react/spend_metrics.jsx';

if (module.hot) {
  module.hot.accept();
}
// import React, { useState, useEffect } from 'react';


// function MetricsVisualization() {

//   const [metricsData, setMetricsData] = useState({}); 

//   useEffect(() => {
//     fetch('/api/metrics-data/')
//       .then(res => res.json())
//       .then(data => {
//         setMetricsData(data);
//       });
//   }, []);

//   // ...
// }

// ReactDOM.render(<DataInputForm />, document.getElementById('app'))

// export async function getModelFields(modelName) {

//     return $.ajax({
//       url: `/api/models/${modelName}/fields`,
//       method: 'GET'  
//     });
  
//   }



const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<HelloWorld />); 

// root.render(<MetricsTracker/>);

root.render(<Add/>)
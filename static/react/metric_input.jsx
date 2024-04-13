// import React, {useState} from 'react';

// const modelOptions = [
//     {label: 'Song Metrics', value: 'song_metrics'},
//     {label: 'Artist Followers', value: 'artist_followers'},
//     {label: 'Money Spent', value: 'money_spent'},
//     {label: 'Money Received', value: 'money_received'},
//     // other models
//   ];
  
//   function ModelSelect({value, onChange}) {
//     return (
//       <select value={value} onChange={onChange}>
//         {modelOptions.map(o => (
//           <option value={o.value}>{o.label}</option>
//         ))}  
//       </select>
//     );
//   }

// function DataInputForm() {

// const [selectedModel, setSelectedModel] = useState();

// function handleModelChange(model) {
//     setSelectedModel(model); 
// }

// return (
//     <div>
    
//     <ModelSelect 
//         value={selectedModel}
//         onChange={handleModelChange}  
//     />

//     <form>
//         {selectedModel && <FieldsForModel model={selectedModel} />}  
//         <button type="submit">Save</button>   
//     </form>

//     </div>
// );
// }

// function FormField({field}) {
//   if (field.type === 'string') {
//     return <input name={field.name} />
//   } else if (field.type === 'int') {
//     return <input type="number" name={field.name} /> 
//   } else if (field.type === 'decimal') {
//     return <input type="number" step="0.01" name={field.name} />
//   } else if (field.type === 'foreignKey') {
//     return <select name={field.name}>{/* options */}</select>
//   }
// }

// export default DataInputForm;

import { useState } from 'react';
import { Line } from 'react-chartjs-2';

const MetricsTracker = () => {
  const [spent, setSpent] = useState([]); 
  const [streams, setStreams] = useState([]);
  const [followers, setFollowers] = useState([]);

  const trackMetrics = e => {
    e.preventDefault();

    const money = e.target.spent.value; 
    const streamCount = e.target.streams.value;
    const followerCount = e.target.followers.value;

    setSpent(prev => [...prev, money]); 
    setStreams(prev => [...prev, streamCount]); 
    setFollowers(prev => [...prev, followerCount]);

    e.target.reset();
  };

  const data = {
    labels: spent.map((_, i) => i),
    datasets: [
      {
        label: 'Money Spent ($)',
        data: spent,
        fill: false,
        backgroundColor: 'rgb(255, 99, 132)',
        borderColor: 'rgba(255, 99, 132, 0.2)',
      },
      {
        label: 'Streams',
        data: streams,
        fill: false,
        backgroundColor: 'rgb(54, 162, 235)',
        borderColor: 'rgba(54, 162, 235, 0.2)',
      },
      {
        label: 'Followers Gained',
        data: followers,
        fill: false,
        backgroundColor: 'rgb(75, 192, 192)',
        borderColor: 'rgba(75, 192, 192, 0.2)',
      },
    ],
  };

  return (
    <div className="App">
      <h2>Metrics Tracker</h2>
      <form onSubmit={trackMetrics}>
        <label>Money Spent: $</label>
        <input name="spent" type="number" step=".01" required/> 
        <label>Streams:</label>
        <input name="streams" type="number" required/>
        <label>Followers Gained:</label>
        <input name="followers" type="number" required/>
        <button>Submit</button>
      </form>
      <Line data={data} />
    </div>
  );
}

export default MetricsTracker;
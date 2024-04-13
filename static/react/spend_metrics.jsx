import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import ReactDOM from 'react-dom';

// function MetricsVisualization({ moneySpent, songData, followersData }) {

//   const [chartData, setChartData] = useState({});

//   useEffect(() => {
//     const labels = moneySpent.map(ms => ms.spent_at);

//     const moneyData = {
//       label: 'Money Spent',
//       data: moneySpent.map(ms => ms.amount) 
//     };

//     const songData = {
//       label: 'Song Streams',
//       data: songData.map(sd => sd.streams)  
//     };

//     const followersData = {
//       label: 'Followers',
//       data: followersData.map(fd => fd.spotify_followers)   
//     };

//     setChartData({
//       labels,
//       datasets: [moneyData, songData, followersData]
//     });

//   }, [moneySpent, songData, followersData]);

//   return (
//     <div>
//       <Line 
//         data={chartData}
//         options={{ maintainAspectRatio: false }}
//       />
//     </div>
//   );
// }

// export default MetricsVisualization;


// function HelloWorld(props) {
//     return <h1>Hello World! Hello {props.name}!!!!!!!!!!!!!!!!!!!</h1>; 
//   }
  
// export default HelloWorld;


export function HelloWorld() {
  return <h1>Hello World!!</h1>;
}


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

export {MetricsTracker};

function Add(x, y) {
  
  return <p>1 + 1</p>;
}

export {Add};
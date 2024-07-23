import React from 'react';
import { createRoot } from 'react-dom/client';
import AlphaTicker from '/static/react/ticker';

const rootElement = document.getElementById('ticker-container');
const root = createRoot(rootElement);

root.render(
  <React.StrictMode>
    <AlphaTicker />
  </React.StrictMode>
);
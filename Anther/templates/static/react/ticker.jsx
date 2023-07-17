import React, { Component } from 'react'
import ReactDOM from 'react-dom';
import Ticker, { FinancialTicker, NewsTicker } from 'nice-react-ticker';

export default class AlphaTicker extends Component {
    render() {
      return (
        <div>
          <div className="newsticker">
            <Ticker isNewsTicker={true} slideSpeed={15}> // helps with setting up animation
              <NewsTicker id="1"  title="This is an APLHA version of Anther CLICK HERE TO RETURN TO HOME PAGE"/>
              <NewsTicker id="3"  title="Join our newsletter for development updates!"/>
              <NewsTicker id="2"  title="This is an APLHA version of Anther CLICK HERE TO RETURN TO HOME PAGE"/>
              <NewsTicker id="4"  title="This is an APLHA version of Anther CLICK HERE TO RETURN TO HOME PAGE"/>
              <NewsTicker id="5"  title="Join our newsletter for development updates!"/>
              <NewsTicker id="6"  title="This is an APLHA version of Anther CLICK HERE TO RETURN TO HOME PAGE"/>
            </Ticker>
          </div>
        </div>
      )
    }
  }

  ReactDOM.render(<AlphaTicker />, document.getElementById('ticker-container'));
const width = innerWidth;
const height = innerHeight;

var svg = d3.select("svg")
    .attr("width",'95%')
    .attr("height",'95%')
    .append('g')
    .attr("transform", "translate("+ width/2 +"," + -50 + ")")


                

var forceStrength = 0.3;
function charge(d) {
        return -forceStrength * d.radius;
      }
    
// Make an AJAX call to grab playlist data from the server
// fetch('/get_playlist_data/')
// d3.json('sandBox_src/playListData.json')
// .then(response => response.json())
// .then(data => {
// // Store the fetched data in a variable or use it as needed
// var playlistData = data;
// playListChart(playlistData)
// })
// .catch(error => {
// console.error("Error fetching playlist data:", error);
// });

d3.json("{% static 'sandBox_src/playListData.json' %}", function(error, data) {
    if (error) {
      console.error("Error loading playlist data:", error);
    } else {
      var playlistData = data;
      playListChart(playlistData);
    }
  });

function playListChart(playlistData) {
    var radiusScale = d3.scaleSqrt()
        .domain([1,300])
        .range([10,80])

    var simulation = d3.forceSimulation()
        .force('boundary', forceBoundary(-width/2+40,0,width/2-40,height+400).border(1).strength(0.0001))
        .force('y', d3.forceY().strength(0.0007).y(height+400))
        .force("collide", d3.forceCollide().strength(0.7).iterations(1).radius(d => d.value))
        .force("charge", d3.forceManyBody().strength(charge))

   
    
        var pl = svg.selectAll('.playlist')
    .data(playlistData)
    .enter().append("circle")
    .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended))
    .attr("class", "playlist")
    .attr("r", function(d) { return radiusScale(d.value); })
    .attr("title", function(d) { return d.name; })
    .attr("id", function(d) { return d.id; })
    .on("click", function(d) {
        var clickedElement = this;
        d3.json("{% static 'sandbox_src/playlist_' %}" + d.id + ".json", function(error, graph_data) {
            if (error) {
                console.error("Error loading graph data:", error);
            } else {
                initializeButton();
                plID = d3.select(clickedElement).attr("id");
                nom = d3.select(clickedElement).data()[0].name;
                console.log(plID);

                initializeSimulation(graph_data);
                initializeDisplay(plID, nom, graph_data);
                initializeForces(graph_data);
            }
        });
        d3.selectAll(".playlist").style("visibility", "hidden");
    })
    .on('mouseover', mouseover)
    .on('mousemove', mousemove)
    .on('mouseout', mouseleave);

    simulation.nodes(playlistData)
        .on('tick', ticked);

    function ticked(){
        pl
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
        simulation.alphaTarget(0.1).restart()
        d3.select('#alpha_value').style('flex-basis', (simulation.alpha()*100) + '%');
    }
};

// Reset to PL view
function initializeButton() {
    var button = d3.select("body")
              .append("button")
              .attr("class", "rtnBtn")
              .text("Playlist View");

    // Set the button's CSS properties
    button.style("position", "absolute")
        .style("top", "5%")
        .style("left", width/2);

    // Toggle from song Graph to playlist view
    button.on("click", function() {
        d3.select('g')
            .attr("transform", "translate("+ width/2 +"," + -50 + ")")

        // Show playlist
        d3.selectAll(".playlist").style("visibility","visible");

        // Hide graph and return button
        d3.selectAll(".rtnBtn").remove();
        d3.selectAll(".nodes").style("visibility","hidden");
        d3.selectAll(".links").style("visibility","hidden");
        d3.selectAll(".playlistName").style("visibility","hidden");
        simulation.alpha(0.5).restart();
    });
}

//*********************************************************//
//*********************  Graph  ***************************//
//*********************************************************//
// svg objects
var link, node;
// the data - an object with nodes and links
var graph;

function sendNodeIdToServer(nodeId) {
    $.ajax({
        type: 'GET',
        url: '/get_tracklist/',
        data: {
            id: nodeId
        },
        success: function (data) {
            // Process the response from the server if needed
            console.log("Server response:", data);
        },
        error: function (error) {
            console.error("Error sending node ID to server:", error);
        }
    });
}

//////////// FORCE SIMULATION //////////// 

// force simulator
var simulation = d3.forceSimulation();

// set up the simulation and event to update locations after each tick
function initializeSimulation(graph_data) {
  simulation.nodes(graph_data.nodes);
  initializeForces(graph_data);
  simulation.on("tick", ticked);
}

// values for all forces
forceProperties = {
    charge: {
        enabled: true,
        strength: -30,
        distanceMin: 1,
        distanceMax: 2000
    },
    collide: {
        enabled: true,
        strength: .7,
        iterations: 1,
        radius: 5
    },
    forceX: {
        enabled: false,
        strength: .1,
        x: .5
    },
    forceY: {
        enabled: false,
        strength: .1,
        y: .5
    },
    link: {
        enabled: true,
        distance: 30,
        iterations: 1
    }
}

// add forces to the simulation
function initializeForces(graph_data) {
    // add forces and associate each with a name
    simulation
        .force("link", d3.forceLink())
        .force("charge", d3.forceManyBody())
        .force("collide", d3.forceCollide())
        .force("forceX", d3.forceX())
        .force("forceY", d3.forceY());
    // apply properties to each of the forces
    updateForces(graph_data);
}

// apply new force properties
function updateForces(graph_data) {
    // get each force by name and update the properties
    simulation.force("charge")
        .strength(forceProperties.charge.strength * forceProperties.charge.enabled)
        .distanceMin(forceProperties.charge.distanceMin)
        .distanceMax(forceProperties.charge.distanceMax);
    simulation.force("collide")
        .strength(forceProperties.collide.strength * forceProperties.collide.enabled)
        .radius(forceProperties.collide.radius)
        .iterations(forceProperties.collide.iterations);
    simulation.force("forceX")
        .strength(forceProperties.forceX.strength * forceProperties.forceX.enabled)
        .x(width * forceProperties.forceX.x);
    simulation.force("forceY")
        .strength(forceProperties.forceY.strength * forceProperties.forceY.enabled)
        .y(height * forceProperties.forceY.y);
    simulation.force("link")
        .id(function(d) {return d.id;})
        .distance(forceProperties.link.distance)
        .iterations(forceProperties.link.iterations)
        .links(forceProperties.link.enabled ? graph_data.links : []);

    // updates ignored until this is run
    // restarts the simulation (important if simulation has already slowed down)
    simulation.alpha(1).restart();
}



//////////// DISPLAY ////////////
// generate the svg objects and force simulation
function initializeDisplay(plID, plName, graph_data) {
    d3.select('g')
        .attr("transform", "translate("+ width/2 +"," + height/2 + ")")
        .append("text")
            .attr("class", "playlistName")
            .attr('y',(height/2)-50)
            .text(plName);

    link = svg.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(graph_data.links)
        .enter().append("line") // Append 'line' elements
        .attr("opacity", "0"); // Apply the 'link' class to each line element

    const labelsContainer = d3.select(".labels-container");

  // set the data and properties of node circles
  node = svg.append("g")
  .attr("class", "nodes")
  .selectAll("circle")
  .data(graph_data.nodes)
  .enter().append("circle")
  .attr("class", "node")
  .attr("r", 5) // Initial radius
  .call(d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended))
  .on('mouseover', function(d) {
      console.log("Mouseover event triggered.");
      console.log("Mouse coordinates: x =", d3.event.pageX, ", y =", d3.event.pageY);
      d3.select(this).transition()
          .duration('100')
          .attr('r', "8")
      let cH = d.id; // Get the ID of the hovered node
      link.filter(function(linkData) {
          return linkData.source.id === cH || linkData.target.id === cH;
        })
          .transition()
          .attr('opacity','1');

      const connectedNodeIds = new Set();

      // Find all nodes connected to the hovered node
      graph_data.links.forEach(linkData => {
          if (linkData.source.id === d.id) {
              connectedNodeIds.add(linkData.target.id);
          } else if (linkData.target.id === d.id) {
              connectedNodeIds.add(linkData.source.id);
          }
      });

      node.transition()
      .style('opacity', function(node) {
          if (connectedNodeIds.has(node.id) || node.id === d.id) {
              return 1; // Set opacity to 1 for connected nodes and hovered node
          } else {
              return 0.2; // Set opacity to 0.2 for non-connected nodes
          }
      });

      // Create a tooltip-like text element
      const nodeLabels = svg.selectAll(".node-label")
      .data([d, ...graph_data.nodes.filter(node => connectedNodeIds.has(node.id))]);
      
      nodeLabels.enter().append("text")
      .attr("class", "node-label")
      .text(function(node) { return node.id; })
      .attr('opacity', '1')
      .attr("fill", "black")
      .attr("transform", function(node) {
          return `translate(${node.x + 10}, ${node.y - 10})`;
      });
  })
  .on('mouseout', function(d, i) {
      d3.select(this).transition()
          .duration('100')
          .attr('r',"5");
      link.transition()
          .attr('opacity','0');
      svg.selectAll(".node-label").remove();
      node.transition().style('opacity', 1);
  });
  // visualize the graph
  updateDisplay();
}

// update the display based on the forces (but not positions)
function updateDisplay() {
    node
        .attr("r", forceProperties.collide.radius)
        .attr("stroke", forceProperties.charge.strength > 0 ? "blue" : "red")
        .attr("stroke-width", forceProperties.charge.enabled==false ? 0 : Math.abs(forceProperties.charge.strength)/15);

}

// update the display positions after each simulation tick
function ticked() {
    link
        .attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node
        .attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });
    d3.select('#alpha_value').style('flex-basis', (simulation.alpha()*100) + '%');
}

//////////// UI EVENTS ////////////

function dragstarted(d) {
    PLtextTooltip.attr("opacity",0)

    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
    }

function dragged(d) {
  d.fx = d3.event.x;
  d.fy = d3.event.y;
}

function dragended(d) {
  if (!d3.event.active) simulation.alphaTarget(0.0001);
  d.fx = null;
  d.fy = null;
}

const PLtextTooltip = svg.append("text")
    .attr('x', 0)
    .attr('y', 0)
    .attr('fill','black')
    .attr('opacity',0);
// MOUSEOVERS

var mouseover = function(d) {
    PLtextTooltip
        .text("Playlist Name: " + d.name)  // Set the tooltip text to the node's title
        .style("transform", `translate(${d.x + 50}px, ${d.y - 10}px)`)  // Use CSS transform for positioning
        .style("opacity", 1);  // Make the tooltip visible

    svg.node().appendChild(PLtextTooltip.node());
  }

var mousemove = function(d) {
    PLtextTooltip
    .style("transform", `translate(${d.x + 10}px, ${d.y - 10}px)`)  // Use CSS transform for positioning
}

var mouseleave = function(d) {
    PLtextTooltip
        .transition()
        .duration(100)
        .style("opacity", 0)

    d3.select(this)
        .style("stroke", "none")
        .style("opacity", 0.8)
}

// update size-related forces
d3.select(window).on("resize", function(){
    let width = +svg.node().getBoundingClientRect().width;
    let height = +svg.node().getBoundingClientRect().height;
    updateForces();
});
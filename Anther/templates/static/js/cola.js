const width = innerWidth;
const height = innerHeight;

var svg = d3.select("svg")
    .attr("width",'95%')
    .attr("height",'95%')
    .append('g')
    .attr("transform", "translate("+ width/2 +"," + -500 + ")")

// Tooltip
const rectWidth = 150;
const rectHeight = 200;
const PLtooltip = svg.append("rect")
    .attr("class","tooltip")
        .attr("x", 100) // position rectangle horizontally
        .attr("y", 100) // position rectangle vertically
        .attr("width", rectWidth)
        .attr("height", rectHeight)
        .attr("stroke", "black")
        .attr("opacity", 0);

const PLtextTooltip = svg.append("text")
                .attr('x', 100)
                .attr('y', 500)
                .attr('fill','green')
                .attr('opacity',0)
                

var forceStrength = 0.03;
function charge(d) {
        return -forceStrength * d.radius;
      }

function playListChart() {
    var radiusScale = d3.scaleSqrt()
        .domain([1,300])
        .range([10,80])

    var simulation = d3.forceSimulation()
        .force('boundary', forceBoundary(-width/2+40,0,width/2-40,height+400).border(1).strength(0.0001))
        .force('y', d3.forceY().strength(0.0003).y(height+400))
        .force("collide", d3.forceCollide().strength(0.7).iterations(1).radius(d => d.value))
        .force("charge", d3.forceManyBody().strength(charge))
 
    d3.queue()
        .defer(d3.json, "/templates/static/sandBox_src/playlistData.json")
        .await(ready)
    
    function ready(error, datapoints){
        var pl = svg.selectAll('.playlist')
            .data(datapoints)
            .enter().append("circle")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .attr("class", "playlist")
            .attr("r", d => radiusScale(d.value))
            .attr("title", d => d.name)
            .attr("id", d => d.id)
            .on("click", function(d){
                initializeButton();
                plID = d3.select(this).attr("id");
                nom = d3.select(this).attr("name");
                console.log(plID);
                d3.selectAll(".playList").style("visibility","hidden")
                d3.json("/templates/static/sandBox_src/miserables.json", function(error, _graph) {
                    if(error) throw error;
                    graph = _graph;
                    initializeDisplay(plID, nom)
                    initializeSimulation()
                });
            })
            .on('mouseover', mouseover)
            .on('mousemove', mousemove)
            .on('mouseout', mouseleave);

        simulation.nodes(datapoints)
            .on('tick', ticked);

        function ticked(){
            pl
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
            simulation.alphaTarget(0.1).restart()
            d3.select('#alpha_value').style('flex-basis', (simulation.alpha()*100) + '%');;
        }
    }
};
playListChart()

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
            .attr("transform", "translate("+ width/2 +"," + -500 + ")")

        // Show playlist
        d3.selectAll(".playList").style("visibility","visible");

        // Hide graph and return button
        d3.selectAll(".rtnBtn").remove();
        d3.selectAll(".nodes").remove();
        d3.selectAll(".links").remove();
        d3.selectAll(".playlistName").remove();
        simulation.alpha(5).restart();
    });
}

//*********************************************************//
//*********************  Graph  ***************************//
//*********************************************************//
// svg objects
var link, node;
// the data - an object with nodes and links
var graph;

//////////// FORCE SIMULATION //////////// 

// force simulator
var simulation = d3.forceSimulation();

// set up the simulation and event to update locations after each tick
function initializeSimulation() {
  simulation.nodes(graph.nodes);
  initializeForces();
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
function initializeForces() {
    // add forces and associate each with a name
    simulation
        .force("link", d3.forceLink())
        .force("charge", d3.forceManyBody())
        .force("collide", d3.forceCollide())
        .force("forceX", d3.forceX())
        .force("forceY", d3.forceY());
    // apply properties to each of the forces
    updateForces();
}

// apply new force properties
function updateForces() {
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
        .links(forceProperties.link.enabled ? graph.links : []);

    // updates ignored until this is run
    // restarts the simulation (important if simulation has already slowed down)
    simulation.alpha(1).restart();
}



//////////// DISPLAY ////////////
// generate the svg objects and force simulation
function initializeDisplay(plID, plName) {
    d3.select('g')
            .attr("transform", "translate("+ width/2 +"," + height/2 + ")")
            .append("text")
                .attr("class", "playlistName")
                .attr('y',(height/2)-50)
                .text(plName)
  // set the data and properties of link lines
  link = svg.append("g")
        .attr("class", "links")
    .selectAll("line")
    .data(graph.links)
    .enter().append("line")
      .attr('opacity', '0');    

  link.append("source")
    .text(function(d) { return d.source; })
  link.append("target")
    .text(function(d) { return d.target; })

    

  // set the data and properties of node circles
  node = svg.append("g")
        .attr("class", "nodes")
    .selectAll("circle")
    .data(graph.nodes)
    .enter().append("circle")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended))
    .on('mouseover', function(d) {
        d3.select(this).transition()
            .duration('100')
            .attr('r', "8")
        let cH = d.id; // Get the ID of the hovered node
        link.filter(function(linkData) {
            return linkData.source.id === cH || linkData.target.id === cH;
          })
            .transition()
            .attr('opacity','1');
    })
    .on('mouseout', function(d, i) {
        d3.select(this).transition()
            .duration('100')
            .attr('r',"5");
        link.transition()
            .attr('opacity','0');
    });

  // node tooltip
  node.append("title")
      .text(function(d) { return d.name; });

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
    PLtooltip.attr("opacity",0)

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

// MOUSEOVERS

var mouseover = function(d) {
    PLtooltip
        .transition()
        .duration(100)
        .style("opacity", 1)
    PLtextTooltip
        .transition()
        .duration(100)
        .style("opacity", 1)
    
    d3.select(this)
      .style("stroke", "black")
      .style("opacity", 1)
  }

var mousemove = function(d) {
    PLtooltip
        .attr("x", (d3.mouse(this)[0]) + "px")
        .attr("y", (d3.mouse(this)[1]) + "px")
    PLtextTooltip
        .text("The exact value of this cell is: " + d.value)
        .attr("x", (d3.mouse(this)[0]) + "px")
        .attr("y", (d3.mouse(this)[1]+20) + "px")
}

var mouseleave = function(d) {
    PLtooltip
        .transition()
        .duration(100)
        .style("opacity", 0)
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
    width = +svg.node().getBoundingClientRect().width;
    height = +svg.node().getBoundingClientRect().height;
    updateForces();
});
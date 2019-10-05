// set the dimensions and margins of the graph
var margin = { top: 10, right: 30, bottom: 30, left: 60 },
   width = 1090 - margin.left - margin.right,
   height = 400 - margin.top - margin.bottom;

// append the svg object to the body of the page
var svg = d3.select("body")
   .append("svg")
   .attr("width", width + margin.left + margin.right)
   .attr("height", height + margin.top + margin.bottom)
   .append("g")
   .attr("transform",
      "translate(" + margin.left + "," + margin.top + ")");

var zoom = d3.zoom()
   .scaleExtent([1, Infinity])
   .translateExtent([[0, 0], [width, height]])
   .extent([[0, 0], [width, height]])
   .on("zoom", zoomed);


var x = d3.scaleTime()
   .range([0, width]);
var x2 = d3.scaleTime().range([0, width]);

var y = d3.scaleLinear()
   .range([height, 0]);


var brush = d3.brushX()
   .extent([[0, 0], [width, height]]);

var xAxis = svg.append("g")
   .attr("class", "axis axis--x")
   .attr("transform", "translate(0," + height + ")");

var xScale = d3.axisBottom(x)
   .ticks(8);

var yAxis = svg.append("g")
   .attr("class", "axis")

// Add a clipPath: everything out of this area won't be drawn.
var clip = svg.append("defs").append("svg:clipPath")
   .attr("id", "clip")
   .append("svg:rect")
   .attr("width", width)
   .attr("height", height)
   .attr("x", 0)
   .attr("y", 0);

var valueLine = d3.line()
   .x(function (d) { return x(d.date) })
   .y(function (d) { return y(d.value) });

// A function that set idleTimeOut to null
var idleTimeout;
function idled() { idleTimeout = null; }


function zoomed() {
   if (d3.event.sourceEvent && d3.event.sourceEvent.type === "brush") return; // ignore zoom-by-brush
   var t = d3.event.transform;
   x.domain(t.rescaleX(x2).domain());

   svg.select(".originalStock").attr("d", valueLine);
   svg.select(".predictedStock").attr("d", valueLine);
   svg.select(".axis--x").call(xScale);


   //svg.select(".brush").call(brush.move, x.range().map(t.invertX, t));

}

function resetted() {
   svg.select(".brush").transition()
      .duration(1000)
      .call(zoom.transform, d3.zoomIdentity);
}


// A function that update the chart for given boundaries
function updateChart() {

   extent = d3.event.selection

   // If no selection, back to initial coordinate. Otherwise, update X axis domain
   if (!extent) {
      if (!idleTimeout) return idleTimeout = setTimeout(idled, 500); // This allows to wait a little bit
      x.domain([4, 8])
   } else {
      x.domain([x.invert(extent[0]), x.invert(extent[1])])
      svg.select(".brush").call(brush.move, null) // This remove the grey brush area as soon as the selection has been done
   }

   // Update axis and line position
   xAxis.transition().duration(1000).call(d3.axisBottom(x));

   svg
      .select('.originalStock')
      .transition()
      .duration(1000)
      .attr("d", valueLine);

   svg
      .select('.predictedStock')
      .transition()
      .duration(1000)
      .attr("d", valueLine);
}

function dbClickHandler(data) {
   x.domain(d3.extent(data, function (d) { return d.date; }))
   xAxis.transition().call(d3.axisBottom(x))
   svg
      .select('.originalStock')
      .transition()
      .attr("d", valueLine);

   svg
      .select('.predictedStock')
      .transition()
      .attr("d", valueLine);
}


function setUpChart(data) {
   x.domain(d3.extent(data, function (d) { return d.date; }));
   x2.domain(x.domain());
   y.domain([0, d3.max(data, function (d) { return +d.value; })])

   xAxis.call(xScale);
   yAxis.call(d3.axisLeft(y));

   // Add brushing
   // Add the brush feature using the d3.brush function
   // initialise the brush area: start at 0,0 and finishes at width,height: it means I select the whole graph area
   brush.on("end", updateChart)               // Each time the brush selection changes, trigger the 'updateChart' function


   d3.select("button")
      .on("click", resetted);

   // Create the line variable: where both the line and the brush take place
   svg.append('g')
      .attr("clip-path", "url(#clip)")
      .append("g")
      .attr("class", "brush")
      .call(brush)
      .call(zoom);

}

function plotStockPrice(fileName, isPredicted) {

   //Read the data
   d3.csv(fileName,
      // When reading the csv, I must format variables:
      function (d) {
         return { date: d3.timeParse("%Y-%m-%d")(d.date), value: +d.price }
      },

      // Now I can use this dataset:
      function (data) {
         if (!isPredicted) {
            setUpChart(data);

            // Add the line
            svg.select('.brush')
               .append("path")
               .datum(data)
               .attr("class", "line originalStock")
               .attr("d", valueLine);

            // If user double click, reinitialize the chart
            svg.on("dblclick", dbClickHandler.bind(this, data));

         } else {
            // Add the line
            svg.select('.brush')
               .append("path")
               .datum(data)
               .attr("class", "line predictedStock")
               .attr("d", valueLine);

            // If user double click, reinitialize the chart
            svg.on("dblclick", dbClickHandler.bind(this, data));
         }
      });
}


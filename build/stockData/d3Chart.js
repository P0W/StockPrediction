/*
* Prashant Srivastava
* Dated: October 22nd, 2019.
*/
// set the dimensions and margins of the graph
var margin = { top: 10, right: 30, bottom: 30, left: 60 },
   width = 850 - margin.left - margin.right,
   height = 400 - margin.top - margin.bottom;

// append the svg object to the body of the page
var svg = d3.select("#CharContainer")
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

var yScale = d3.axisLeft(y);

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

var bisectDate = d3.bisector(function (d) { return d.date; }).left;
var timeParser = d3.timeParse("%Y-%m-%d");

// A function that set idleTimeOut to null
var idleTimeout, chartArea, cursor, stockData = null, currentSelectedStock;
function idled() { idleTimeout = null; }

// Create the line variable: where both the line and the brush take place
chartArea = svg.append('g')
   .attr("clip-path", "url(#clip)")
   .append("g")
   .attr("class", "brush");

function clearGraph() {
   x.domain([0, 0]);
   y.domain([0, 0]);
   xAxis.call(xScale);
   yAxis.call(yScale);
   svg.selectAll("path").remove();
}

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
   chartArea.transition()
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

function setDomain(newData) {
   x.domain(d3.extent(newData, function (d) {
      return d.date;
   }));
   x2.domain(x.domain());

   y.domain([0, d3.max(newData, function (d) {
      return +d.value;
   })]);

   xAxis.call(xScale);
   yAxis.call(yScale);
}

function resetDomain(newData) {
   var currDomain = x.domain();
   x.domain([Math.min(currDomain[0], newData[0].date), Math.max(currDomain[1], newData[newData.length - 1].date)]);
   x2.domain(x.domain());

   var currDomain = y.domain();
   y.domain([0, Math.max(currDomain[1], d3.max(newData, function (d) { return +d.value; }))]);

   xAxis.call(xScale);
   yAxis.call(yScale);
}

function setUpChart() {
   // Add brushing
   // Add the brush feature using the d3.brush function
   // initialise the brush area: start at 0,0 and finishes at width,height: it means I select the whole graph area
   brush.on("end", updateChart);// Each time the brush selection changes, trigger the 'updateChart' function


   d3.select("#ResetButton")
      .on("click", resetted);

   chartArea
      .call(brush)
      .call(zoom);

   var x1 = x;

   var marker = svg.append('circle')
      .attr('class', 'marker');

   cursor = chartArea.append('line')
      .attr('class', 'cursor')
      .style('display', 'none')
      .attr('x1', margin.left)
      .attr('y1', 0)
      .attr('x2', margin.left)
      .attr('y2', height);

   chartArea.on('mousemove', function () {
      var mouse = d3.mouse(this);
      cursor.style('display', 'block');
      var mouseDate = x1.invert(mouse[0]);
      var i = bisectDate(stockData, mouseDate);
      if (i <= 0) return;

      var d0 = stockData[i - 1];
      var d = d0;
      var xPos = x1(d.date);
      var yPos = y(d.value);
      cursor
         .attr('x1', xPos)
         .attr('x2', xPos);

      var svgPositon = d3.select('svg').node().getBoundingClientRect();
      d3.select('.tooltip')
         .text(d.value + ' on ' + d3.timeFormat('%Y-%b-%d')(d.date))
         .style('display', 'block')
         .style('left', xPos + svgPositon.x + 'px')
         .style('top', yPos + 'px');

      marker.attr('cx', xPos)
         .attr('cy', yPos)
         .attr('r', 3.5);

   })
      .on('mouseout', function () {
         cursor.style('display', 'none');
         d3.select('.tooltip').style('display', 'none');
         marker.attr('r', 0.0);
      })
      .on('mouseover', function () {
         cursor.style('display', null);
         d3.select('.tooltip').style('display', 'null');
         marker.attr('r', 0.0);
      })

}

function joinAllStockData(newData) {
   var temStockData = {};

   if (stockData !== null) {
      stockData.forEach(function (s) {
         var key = d3.timeFormat('%s')(s.date);
         temStockData[key] = s.value;
      });
   }

   newData.forEach(function (s) {
      var key = d3.timeFormat('%s')(s.date);
      temStockData[key] = s.value;
   });

   stockData = [];
   Object.keys(temStockData).forEach(function (d) {
      stockData.push({
         'date': d3.timeParse("%s")(d),
         'value': temStockData[d]
      })
   });
}

function getLastDate() {
   if (stockData !== null) {
      return stockData[stockData.length - 1].date;
   } else {
      return new Date();
   }
}

function showpredictions(actual, predicted) {
   stockData = [];
   joinAllStockData(actual);

   if (typeof predicted !== 'undefined') {
      joinAllStockData(predicted);
      stockData = actual.concat(predicted);
   }

   stockData.sort((a, b) => {
      return a.date - b.date;
   })
   setDomain(stockData);
   //resetDomain(predicted);

   // Add the line
   svg.select('.brush')
      .append("path")
      .datum(actual)
      .attr("class", "line originalStock")
      .attr("d", valueLine);


   // Add the line
   if (typeof predicted !== 'undefined') {
      svg.select('.brush')
         .append("path")
         .datum(predicted)
         .attr("class", "line predictedStock")
         .attr("d", valueLine);
   }
   setUpChart();
}

function plotTestData(stockSymbol) {
   var promiseObj = new Promise(resolve => {
      d3.csv(stockSymbol + '_test_pred.csv',
         // When reading the csv, I must format variables:
         function (d) {
            return {
               date: timeParser(d.date),
               actual_value: +d.actual_price,
               predicted_value: +d.price
            };
         },

         // Now I can use this dataset:
         function (dataset) {
            var actualTestData = [];
            var predictedTestData = [];
            dataset.forEach(item => {
               actualTestData.push({
                  date: item.date,
                  value: item.actual_value
               });
               predictedTestData.push({
                  date: item.date,
                  value: item.predicted_value
               });
            });

            clearGraph();
            showpredictions(actualTestData, predictedTestData);
            resolve('resolved');

         });
   });
   return promiseObj;
}

function setSelectedvalue(selectedStock) {
   currentSelectedStock = selectedStock;
}

// Function to request data from frontend
function fetchData(args) {
   // The REST like URL
   var url = "http://localhost:4242/stockData/" + currentSelectedStock + "?" + args;
   // Use the built-in XMLHttpRequest from C++ backend
   fetch(url)
      // Convert reqsponse object to text
      .then(response => response.text())
      // When contents are recieved do the required processing
      .then(contents => {
         // Check if its test or train data
         if (args === 'testData' || args === 'trainData') {
            plotTestData(currentSelectedStock);
         } else {
            // if  not train or test request it is request for future price prediction
            createFuturePrices(currentSelectedStock);
         }
         // Plot the data for visualization
         d3.select("#loader").style('display', 'none');
      })
      // Handle any errors, if any
      .catch(() => console.log("Cannot access " + url + " Blocked by browser ?"));
}

function getNextDay(from, N) {
   var nextDay = new Date(from);
   nextDay.setDate(nextDay.getDate() + N);
   return nextDay;
}

function createFuturePrices(stockSymbol) {

   var futureDataSet = [];
   //Read the data
   d3.csv(stockSymbol + '_future.csv',
      // When reading the csv, I must format variables:
      function (d) {
         return { value: +d.price }
      },
      // Now I can use this dataset:
      function (dataset) {
         var lastDate = getLastDate();
         dataset.forEach((element, index) => {
            futureDataSet.push({
               'date': getNextDay(lastDate, index),
               'value': element.value
            })
         });
         clearGraph();
         showpredictions(futureDataSet);
         setUpChart();
      });
}
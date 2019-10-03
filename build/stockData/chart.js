
// set the dimensions and margins of the graph
var margin = { top: 20, right: 20, bottom: 70, left: 50 },
    width = 1570 - margin.left - margin.right,
    height = 690 - margin.top - margin.bottom;

// parse the date / time
var parseTime = d3.timeParse("%Y-%d-%d");

// set the ranges
var x = d3.scaleLinear().range([0, width]).nice();
var y = d3.scaleLinear().range([height, 0]).nice();

var xTime = d3.scaleTime().range([0, width]).nice();

// define the line
var valueline = d3.line()
    .x(function (d) { return x(d.index); })
    .y(function (d) { return y(d.price); })
    .curve(d3.curveLinear)

// append the svg obgect to the body of the page
// appends a 'group' element to 'svg'
// moves the 'group' element to the top left margin
var svg = d3.select("body").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");


// Get the data

function setUpPlot(data) {

    // Scale the range of the data
    xTime.domain(d3.extent(data, function (d) { return d.time; }));
    // Add the X Axis
    svg.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(xTime).ticks(20))
        .selectAll("text")
        .style("text-anchor", "end")
        .attr("dx", "-.8em")
        .attr("dy", ".15em")
        .attr("transform", "rotate(-65)");

    // Add the Y Axis
    svg.append("g")
        .attr("class", "axis")
        .call(d3.axisLeft(y).ticks(10));

}

function plotStockPrice(csvName, isPredicted) {
    d3.csv(csvName, function (error, data) {
        if (error) throw error;

        // format the data
        data.forEach(function (d, index) {
            d.time = parseTime(d.date),
                d.price = +d.price,
                d.index = index
        });
        if (typeof isPredicted !== 'undefined' && isPredicted) {
            // Add the valueline for Predicted Stock Prices path.
            svg.append("path")
                .data([data])
                .attr("class", "line")
                .style("stroke", "red")
                .attr("d", valueline);
        } else {
            // Add the valueline for Actual Stock prices path.
            y.domain([0.0, d3.max(data, function (d) { return d.price; })]);
            setUpPlot(data);
            x.domain([1.0, data.length]);

            svg.append("path")
                .data([data])
                .attr("class", "line")
                .style("stroke", "blue")
                .attr("d", valueline);
        }


    });
}
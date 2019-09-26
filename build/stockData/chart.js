
// set the dimensions and margins of the graph
var margin = { top: 20, right: 20, bottom: 70, left: 50 },
    width = 1580 - margin.left - margin.right,
    height = 600 - margin.top - margin.bottom;

// parse the date / time
var parseTime = d3.timeParse("%Y-%d-%d");

// set the ranges
var x = d3.scaleTime().range([0, width]).nice();
var y = d3.scaleLinear().range([height, 0]).nice();

// define the line
var valueline = d3.line()
    .x(function (d) { return x(d.date); })
    .y(function (d) { return y(d.price); })
//.curve(d3.curveBasis);

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
function plotStockPrice(csvName) {
    d3.csv(csvName, function (error, data) {
        if (error) throw error;

        // format the data
        data.forEach(function (d) {
            d.date = parseTime(d.date);
            d.price = +d.price;
        });

        // Scale the range of the data
        x.domain(d3.extent(data, function (d) { return d.date; }));
        y.domain([0.0, d3.max(data, function (d) { return d.price; })]);

        // Add the valueline path.
        svg.append("path")
            .data([data])
            .attr("class", "line")
            .attr("d", valueline);

        // Add the X Axis
        svg.append("g")
            .attr("class", "axis")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x).ticks(10))
            .selectAll("text")
            .style("text-anchor", "end")
            .attr("dx", "-.8em")
            .attr("dy", ".15em")
            .attr("transform", "rotate(-65)");

        // Add the Y Axis
        svg.append("g")
            .attr("class", "axis")
            .call(d3.axisLeft(y).ticks(10));

    });
}
# bolt_lanes_polyfit
<p>This repsoitory contains code for converting the lane masked data which is obtained from the yolop algorithm to quadratic polynomials in the world frame using many algorithms like Pinhole Camera Model, Dbscan. The clustering of lanes works prettly well and we can also get ploynomials from it. The basic tracking algorithm works on the markov first order principle, but still only improves the lane tracking by a little.</p>

### The algorithm works well when we get really good lane data , which can be observed from the following



https://github.com/KeerthivasanIITMadras/bolt_lanes_polyfit/assets/94305617/171c2bdc-f114-41ec-9800-3d58d4b5239a

<p>In this test initially it fails due to bad lane data but after we get good lanes it works really well</p>


https://github.com/KeerthivasanIITMadras/bolt_lanes_polyfit/assets/94305617/68463ad3-9b20-4c37-8b6c-22093ff20c8c


var map;
function initMap() {
  map = new google.maps.Map(document.getElementById('mapid'), {
  center: new google.maps.LatLng(33.4890, 126.4983),
  zoom: 10
  });
}

window.marker_callback = function(markersdata) {
    for(var i=0; i< markersdata.length; i++) {
        var current = markersdata[i];
        var point = new google.maps.LatLng(
                  parseFloat(current.coordinates[0]),
                  parseFloat(current.coordinates[1]));

        console.log("here"+point)

              infowincontent = document.createElement('div');
              var strong = document.createElement('strong');
              strong.textContent =current.title;
              infowincontent.appendChild(strong);
              infowincontent.appendChild(document.createElement('br'));

              var text = document.createElement('text');
              text.textContent = "ID:"+current.id+"URL:"+current.url;
              infowincontent.appendChild(text);
              //var icon = customLabel[type] || {};


            var marker = new google.maps.Marker({
                map: map,
                position: point,
                icon:"http://chart.apis.google.com/chart?chst=d_map_pin_letter&chld=%E2%80%A2|"+color,
                title:type,
                animation:google.maps.Animation.DROP,
                draggable: true
                //label: icon.label
              });

            google.maps.event.addListener(marker, 'click', function () {
            // where I have added .html to the marker object.
              infoWindowOnClick.setContent(infowincontent);
              infoWindowOnClick.open(map, this);
            });
    }

}


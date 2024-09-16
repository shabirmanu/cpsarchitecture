





$(function() {

      // Modal Bind Functions
    $('#taskModal').on('hide.bs.modal show.bs.modal', function(event) {
        var $activeElement = $(document.activeElement);

  if ($activeElement.is('[data-toggle], [data-dismiss]')) {
        if (event.type === 'hide') {
      // Do something with the button that closed the modal
            console.log('The button that closed the modal is: ', $activeElement);

        }

    if (event.type === 'show') {
      // Do something with the button that opened the modal
      console.log('The button that opened the modal is: ', $activeElement);
      $('#discardTask').on('click', function () {
          console.log("Discard button clicked")
          $activeElement.closest('tr').remove()
          $('#taskModal').modal('toggle');

          // $.ajax({
          //       type: "POST",
          //       taskName: currentTask,
          //       url: discardURL,
          //       flag:tFlag,
          //       success: function (resp) {
          //           var htmMark = '';
          //           for(var i =0; i<resp.length; i++) {
          //
          //               var ind = resp[i].toString().split(",")[0];
          //               var val = resp[i].toString().split(",")[1];
          //
          //               htmMark += "<option value="+ind+">"+val+"</option>";
          //           }
          //           //dropdown.mserv.attr('enabled', 'enabled');
          //           $('#select_mservice').html(htmMark)
          //       }
          //   });

      });
    }
  }
});

    // $('.spinner').hide();
    // $('a.deploy').bind('click', function() {
    //     $(this).text('Deploying');
    //     url = $(this).attr('href');
    //     console.log("this url==="+url);
    //      URLArr = url.split('?');
    //      mainURL = URLArr[0];
    //      task = URLArr[1].split('=')[1];
    //      console.log(task)
    //
    //     console.log("main   "+mainURL)
    //      id = $(this).attr('id');
    //      $('#spinner-'+id).show();
    //     $.ajax({
    //           type: "GET",
    //           dataType: 'json',
    //           data: {'id': id, 'task':task},
    //           crossDomain:true,
    //           url: mainURL
    //     }).done(function (data) {
    //         $('#spinner-'+id).hide();
    //
    //         if(data) {
    //             console.log(data[0].timestamp);
    //             $('#spinner-'+id).parent().next().html("Reading Time: "+data[0].timestamp+"<br />Sensor Value: "+data[1].sensor_val);
    //             $('#'+id).text('Deployed');
    //             $('#'+id).addClass('btn-danger')
    //             $('#'+id).attr('href','#');
    //         }
    //     }).fail(function(data) {
    //
    //         $('#spinner-'+id).parent().next().html("Error Accessing IoT Gateway!");
    //     });


        // console.log(url)
        // console.log(id)
        //
        //
        // $.get(url, {id: id}, function(data) {
        // $("#result").text(data.result);
        // });

        return false;
    });



    // jQuery selection for the 2 select boxes
    var dropdown = {
        service: $('#select_service'),
        mserv: $('#select_mservice')
    };

    // call to update on load
    updateMicroservices();

    // function to call XHR and update county dropdown
    function updateMicroservices() {
        var send = {
            service: dropdown.service.val()
        };
        //dropdown.mserv.attr('disabled', 'disabled');
        dropdown.mserv.empty();

       // $("#select_service").change(function () {
            //let user_identifier = this.value;
        console.log(send)
        $.ajax({
                type: "GET",
                url:url_val,
                data: send,
                success: function (resp) {
                    var htmMark = '';
                    for(var i =0; i<resp.length; i++) {

                        var ind = resp[i].toString().split(",")[0];
                        var val = resp[i].toString().split(",")[1];

                        htmMark += "<option value="+ind+">"+val+"</option>";
                    }
                    //dropdown.mserv.attr('enabled', 'enabled');
                    $('#select_mservice').html(htmMark)
                }
            });
       // });





    }

    // event listener to state dropdown change
    dropdown.service.on('change', function() {
        console.log("changed");
        updateMicroservices();
    });




  });


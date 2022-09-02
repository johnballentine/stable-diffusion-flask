$( document ).ready(function() {

  $("#generateBtn").click( function() {

    jsonData = {
      "prompt": $('#promptTxt').val()
    };

    // Remove last generation if it exists
    if ($('#resultImg').length) {
      $('#resultImg').remove();
    }

    $('#loader').show();

    $.ajax({
      url: '/txt2img/endpoint',
      dataType: 'text',
      type: 'post',
      contentType: 'application/json',
      data: JSON.stringify(jsonData),
      success: function(data){
        $('#loader').hide();
        var img = $('<img id="resultImg" class="resultImg">');
        img.attr('src', data);
        img.appendTo('#displayDiv');
      },
      error: function (xhr, ajaxOptions, thrownError) {
        $('#loader').hide();
        console.log("Error posting to endpoint.");
        console.log(xhr.status);
        console.log(thrownError);
      }
      });
  });

});

$(document).ready(function() {

  var wikiApiUrl = 'parse_sql';
  var template = '<div class="card result"><h2 class="title">{{title}}</h2><div class="snippet">{{snippet}}</div><a href="{{url}}" class="read-more-link">Read More</a></div>';

  //handels error to update ui
  function showError(error) {
    $('.error').text(error).removeClass('zoomOutRight').addClass('slideInRight').show();
    setTimeout(function() {
      $('.error').removeClass('slideInRight').addClass('zoomOutRight').hide(1000).text('');
    }, 3000);
  }


  function wikiRequest(query) {
      $.ajax({
            type: 'GET',
            url: wikiApiUrl+query,
            dataType: 'jsonp', //希望服务器返回json格式的数据
            jsonp: "callback",
            jsonpCallback: "successCallback",
            success: function (data) {
                //console.log(data[0]['prediction']);
              var el = '<div class="card result animated fadeIn"><h2 class="title">' + data[0]['org'] + '</h2><div class="snippet">' + data[0]['title'] + '</div><a class="read-more-link" target="_blank">'+data[0]['prediction'] +'</a></div>';
          // console.log(el);
             $('.content').append(el);
            }
      })
    }

  function submit_sync(query,table_id) {
    $.ajax({
        type: "post",
        url: wikiApiUrl,
        async: false,
        data: JSON.stringify({
            question: query,
            table_id: table_id
        }),
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function(data) {
            $('#fade_id').remove();
            console.log(data)
            var el = '<div class="card result animated fadeIn" id="fade_id"><h2 class="title">' + data['task'] + '</h2></div>';
          // console.log(el);
             $('.content').append(el);
        } // 注意不要在此行增加逗号
    })}



  //listens for clear button click of search input and removes existing results
  $('#wiki-search').on('search', function(e) {
    var query = $('#wiki-search').val();

    if (!query) {
      showError('Type something to search!..');
      return;
    }
  });


  $('#random-article').on('click', function(e) {
     $.get("http://127.0.0.1:8080/random-article", function(result){
        $('#wiki-search').val(result)
        $('#wiki-search')[0].focus();

  })
   });


  //Written before autocomplete plugin added , it will call wikiRequest() when user press Enter in searcg bar.
  $('#wiki-search-form').on('submit', function(e) {

    e.preventDefault();


    var query = $('#wiki-search').val();
    var e = document.getElementById("select_box");
    var table_id = e.options[e.selectedIndex].value;

    if (!query) {
      showError('Type something to search!..');
      return;
    }else{
      console.log(query);
      console.log(table_id);
      submit_sync(query,table_id);

    }

  })



  //if user deletes text in search feild then existing results are removed
  $('#wiki-search').on('change', function() {
    if (this.value == "") {
      $('.result').remove();
    }
  });

});

$(document).ready(function() {

  var wikiApiUrl = 'http://127.0.0.1:8080/content_article/';
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
              var el = '<div class="card result animated fadeIn"><h2 class="title">'  + '</h2><div class="snippet">' + data[0]['article'] + '</div><a class="read-more-link" target="_blank">'+data[0]['prediction'] +'</a></div>';
          // console.log(el);
             $('.content').append(el);
            }
      })
    }

  //listens for clear button click of search input and removes existing results
  $('#wiki-search').on('search', function(e) {
    if (this.value == "") {
      $('.result').remove();
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
    if (!query) {
      showError('Type something to search!..');
      return;
    }else{
      console.log(query);
      wikiRequest(query);
    }

  })



  //if user deletes text in search feild then existing results are removed
  $('#wiki-search').on('change', function() {
    if (this.value == "") {
      $('.result').remove();
    }
  });

});

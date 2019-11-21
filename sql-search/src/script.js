$(document).ready(function() {

  var wikiApiUrl = '//en.wikipedia.org/w/api.php';
  var template = '<div class="card result"><h2 class="title">{{title}}</h2><div class="snippet">{{snippet}}</div><a href="{{url}}" class="read-more-link">Read More</a></div>';

  //handels error to update ui
  function showError(error) {
    $('.error').text(error).removeClass('zoomOutRight').addClass('slideInRight').show();
    setTimeout(function() {
      $('.error').removeClass('slideInRight').addClass('zoomOutRight').hide(1000).text('');
    }, 3000);
  }

  //sends ajax request to wiki api to get results
  function wikiRequest(query) {
    $.ajax({
      url: wikiApiUrl,
      data: {
        action: 'opensearch',
        search: query,
        format: 'json'
      },
      dataType: 'jsonp',
      beforeSend: function(xhr) {
        console.log('Before: ' + xhr);
        $('.result').remove();
        $('.content .loading').show();
      },
      complete: function(xhr, status) {
        console.log('complete: ' + status);
        $('.content .loading').hide();
      },
      success: function(result, status, xhr) {
        console.log(result);

        //console.log(resultArr);
        for (var i = 0; i < result[1].length; i++) {
          var title = result[1][i];
          var snippet = result[2][i];
          var url = result[3][i];
          var el = '<div class="card result animated fadeIn"><h2 class="title">' + title + '</h2><div class="snippet">' + snippet + '</div><a href="' + url + '"class="read-more-link" target="_blank">Read More</a></div>';
          // console.log(el);
          $('.content').append(el);
        }
      },
      error: function(xhr, status, error) {
        console.log(error);
      }
    });
  }

  //listens for clear button click of search input and removes existing results
  $('#wiki-search').on('search', function(e) {
    if (this.value == "") {
      $('.result').remove();
    }
  });

  //Written before autocomplete plugin added , it will call wikiRequest() when user press Enter in searcg bar.
  $('#wiki-search-form').on('submit', function(e) {
    e.preventDefault();

    var query = $('#wiki-search').val();
    if (!query) {
      showError('Type something to search!..');
      return;
    }else{
      wikiRequest(query);
    }

  })

  //handels autocomplete
  $('#wiki-search').autocomplete({
    lookup: function(query, done) {
      $.ajax({
        url: wikiApiUrl,
        data: {
          action: 'query',
          list: 'search',
          srsearch: query,
          format: 'json'
        },
        dataType: 'jsonp',

        success: function(data) {
          console.log(data);
          var results = data.query.search
          results.sort(function(a, b) {
            return a.title.localeCompare(b.title);
          })

          data = {
            suggestions: results.map(function(item, index) {
              return {
                value: item.title,
                data: index
              };
            })
          };
          done(data);
        }
      });
    },
    onSelect: function(suggestion) {
      wikiRequest(suggestion.value);
    }
  });

  //if user deletes text in search feild then existing results are removed
  $('#wiki-search').on('change', function() {
    if (this.value == "") {
      $('.result').remove();
    }
  });

});

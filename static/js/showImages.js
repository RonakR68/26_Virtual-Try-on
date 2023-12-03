function showImages(images) {
        console.log("inside showImages");
        for(var i=0;i<images.length;i++)
        {
        /*console.log(images[i]);
        }*/

                //var image1 = 'https://image.flaticon.com/teams/slug/freepik.jpg';
                var html = [
                '<div class="uicomponent-panel-controls-container">',
                '<img src=' + images[i] + '>',
                '</div>'
                ].join('\n');
                //document.getElementById("dock").innerHTML += html;
        }

        }
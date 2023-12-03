function chooseFile() {
    $('#searchId').val("");
    document.getElementById("fileInput").click();

    // $('#searchId').val($('#fileInput')[0].files[0].name)
}

$('input[type=file]').change(function () {
    $('#searchId').val($('#fileInput')[0].files[0].name)
});


function getSimilarWatches() {


    if ($('#fileInput')[0].files[0] == null) {
        alert("Please enter your watch details or select a similar image")

    } else {
        var formData = new FormData();
        formData.append('file', $('#fileInput')[0].files[0]);



        $.ajax({
            url: 'http://10.0.0.129:8080/Predict/findSimilar',
            type: 'POST',
            data: formData,
            processData: false,  // tell jQuery not to process the data
            contentType: false,  // tell jQuery not to set contentType
            crossDomain:true,
            crossOrigin: true,
            success: function (data) {
                console.log(data);
                // alert(data);
                sessionStorage.setItem("similarWatches", JSON.stringify(data));

                console.log("Openeing page: ")
                window.open("watchList.html")


            }
        });
    }


}


function createWatchItem(watchItem) {

    /**
     *  <div class="card">
     <img src="jeans3.jpg" alt="Denim Jeans" style="width:100%">
     <h1>Tailored Jeans</h1>
     <p class="price">$19.99</p>
     <p>Some text about the jeans..</p>
     <p><button>Add to Cart</button></p>
     </div>
     */
    console.log("Creatimg div for ", watchItem)

    var div = [`<div class="watchCard">`,
        `<img src="` + watchItem['imagePath'] + `" style="object-fit: scale-down;width: 100px;height: 100px;"">`,
        `<h5>` + watchItem['brand'] + `</h5>`,
        `<h6>` + watchItem['name'] + `</h6>`,
        `<p class="price">$$.$$</p>`,
        // `<p>` + watchItem['description'] + `</p>`,
        `</div>`
    ].join("\n");

    return div;

}

function populateWatchInfos() {



    console.log("Populating watches...");

    watchDetails = JSON.parse(sessionStorage.getItem("similarWatches"));
    console.log("watchDetails: ", watchDetails);

    var container = `<div class="container">`;
    var sRow = `<div class="row">`;
    var eDiv = `</div>`;
    var scol = `<div class="col-md-3 col-sm-6">`;
    var watchList = "";


    var index = 0;

    for (let row = 0; row < watchDetails.length / 4; row++) {
        var items = [sRow];
        for (let i = 0; i < 4 && index < watchDetails.length; i++, index++) {
            items.push(scol);
            items.push(createWatchItem(watchDetails[index]));
            items.push(eDiv);
        }
        items.push(eDiv);
        console.log("items", items);
        watchList = watchList + items.join("\n");
    }
    watchList.concat(eDiv);

    // for (let i = 0; i < watchDetails.length; i++) {
    //     watchItem = watchDetails[i];
    //     watchList = watchList + createWatchItem(watchItem);
    // }

    // watchList = watchList + "</ul>";

    console.log("Generated list...", watchList);
    return watchList;


}

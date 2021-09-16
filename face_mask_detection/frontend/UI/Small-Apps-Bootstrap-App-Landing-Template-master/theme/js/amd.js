let detection_model_video, classification_model_video, ctx, videoWidth, videoHeight, video, canvas;

async function setupCamera() {
  video = document.getElementById('videoElement');

  const constraints = {
    audio: false,
    video: true
  };

  function handleSuccess(stream) {
    window.stream = stream; // make stream available to browser console
    video.srcObject = stream;
  }

  function handleError(error) {
    console.log('navigator.MediaDevices.getUserMedia error: ', error.message, error.name);
  }

  const stream = await navigator.mediaDevices.getUserMedia(constraints).then(handleSuccess).catch(handleError);

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const preprocessPrediction = (prediction) => {
  const start = prediction.topLeft;
  const end = prediction.bottomRight;
  x = start[0]
  y = start[1]
  width = start[0] - end[0]
  height = start[1] - end[1]
  diff = 0

  if (height < width) {

    const delta = parseInt(Math.round((height - width) / 2))
    const y_min = y - diff - delta
    const y_max = y + height + diff
    const x_min = x - delta - diff
    const x_max = x + width + delta + diff

    let width_ = x_min - x_max
    let height_ = y_min - y_max
    const width_delta = width_ / 3
    const height_delta = height_ / 5

    const x_ = start[0] + delta + width_delta / 2
    const y_ = start[1]
    width_ = width_ - width_delta
    height_ = height_ - height_delta
    return [x_, y_, width_, height_]
  } else if (width < height) {

    const delta = parseInt(Math.round((width - height) / 2))
    const y_min = y - delta - diff
    const y_max = y + height + delta + diff
    const x_min = x - diff
    const x_max = x + width + diff

    let width_ = x_min - x_max
    const width_delta = width_ / 3

    const x_ = start[0] + width_delta / 2
    const y_ = start[1] + delta
    width_ = width_ - width_delta
    const height_ = y_min - y_max
    return [x_, y_, width_, height_]
  }
}

const renderPrediction = async () => {
  tf.engine().startScope()
  const returnTensors = false;
  const flipHorizontal = false;
  const annotateBoxes = false;
  const predictions = await detection_model_video.estimateFaces(video, returnTensors, flipHorizontal, annotateBoxes);

  if (predictions.length > 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < predictions.length; i++) {
      if (returnTensors) {
        predictions[i].topLeft = predictions[i].topLeft.arraySync();
        predictions[i].bottomRight = predictions[i].bottomRight.arraySync();
      }

      let x_, y_, width_, height_
      [x_, y_, width_, height_] = preprocessPrediction(predictions[i]);

      const img_width = video.width
      const img_height = video.height

      const x_normed = (x_ - 50)/ img_width
      const y_normed = (y_ -50) / img_height
      const width_normed = (width_ + 75) / img_width
      const height_normed = (height_ + 75) / img_height

      const img_tensor = tf.browser.fromPixels(video)

      const reshaped = img_tensor.reshape([1, Math.floor(img_height), Math.floor(img_width), 3])
      const resized = tf.image.cropAndResize(reshaped, [[y_normed, x_normed, y_normed + height_normed, x_normed + width_normed]], [0], [224, 224])
      const normed = resized.div(255.0)
      const prediction = classification_model_video.predict(normed).dataSync()
      tf.dispose()
      prediction_class = prediction.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
      predictions[i].predictionClass = prediction_class
      // console.log(prediction)
      var color = ""
      var title = ""
      if (prediction_class == 0) {
        color = "rgba(255, 0, 0, 1)"
        title = "No mask"
      } else if (prediction_class == 1) {
        color = "rgba(255, 165, 0, 1)"
        title = "Bad mask"
      } else {
        color = "#2ecc71"
        title = "OK mask"
      }
      title = title + " - " + prediction[prediction_class].toFixed(2);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = "2";
      ctx.strokeRect(x_, y_, width_, height_);

      var width_title = ctx.measureText(title).width;

      ctx.fillStyle = color;
      ctx.fillRect(x_ - 1, y_ - 17, width_title + 10, 17);
      ctx.fillStyle = "#FFF";
      ctx.font = "13px Helvetica";
      ctx.fillText(title, x_ + 5, y_ - 5);
    }
    canvas.data = predictions.map((prediction) => preprocessPrediction(prediction).concat([prediction.predictionClass]))
  }
  tf.engine().endScope()
  requestAnimationFrame(renderPrediction);
};

const state = {
  backend: 'webgl'
};

const setupPage = async () => {
  await setupCamera();
  video.play();

  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  canvas = document.getElementById('videoCanvas');

  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctx = canvas.getContext('2d');
  ctx.fillStyle = "rgba(255, 0, 0, 0.5)";

  renderPrediction();
};

async function load_detection_model() {
  detection_model_video = await blazeface.load();
}

async function load_classification_model() {
  classification_model_video = await tf.loadLayersModel("js/model.json");
}

Promise.all([load_detection_model(), load_classification_model()])

function dataURItoBlob(dataURI) {
  // convert base64 to raw binary data held in a string
  // doesn't handle URLEncoded DataURIs - see SO answer #6850276 for code that does this
  var byteString = atob(dataURI.split(',')[1]);

  // separate out the mime component
  var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0]

  // write the bytes of the string to an ArrayBuffer
  var ab = new ArrayBuffer(byteString.length);

  // create a view into the buffer
  var ia = new Uint8Array(ab);

  // set the bytes of the buffer to the correct values
  for (var i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
  }

  // write the ArrayBuffer to a blob, and you're done
  var blob = new Blob([ab], {type: mimeString});
  return blob;

}


function capture() {
  var canvas = document.getElementById('canvas');
  var video = document.getElementById('videoElement');
  var videoCanvas = document.getElementById('videoCanvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const predictions = videoCanvas.data

  const formData = new FormData();
  const fileField = document.querySelector('input[type="file"]');
  const img_tensor = tf.browser.fromPixels(video);

  // wrap around a timer ??!!?!
  predictions.forEach((prediction,index) => {
    if (prediction[4] === 0) {
      var height_preview = prediction[3]
      var width_preview = prediction[2]
      var ctx = canvas.getContext('2d')
      ctx.drawImage(video, prediction[0]-0.5*width_preview, prediction[1]-0.25*height_preview, 2*width_preview, 1.5*height_preview, index*100, 0, 100, 140);
      var email = document.getElementById('email').value;


      var image = new Image();
      image.id = "test-1";
      image.src = canvas.toDataURL();
      //document.getElementById('img-to-crop').appendChild(image);
      
      
      blob = dataURItoBlob(canvas.toDataURL())
      console.log("Blob", blob)

      //console.log("src", canvas.toDataURL())
      //formData.append('frame', canvas.toDataURL('image/jpeg', 0.5))
      formData.append('blob', blob)
      console.log("form data", formData);

      formData.append('email', email)

      console.log("email", email)
      console.log("formData", formData)



    }
  })

  fetch("https://amdemail.herokuapp.com/files", {
    method: "POST",
    body: formData,
    //headers: {'content-type': 'application/x-www-form-urlencoded'}

  })
  .then(response => response.json())
  .then(result => {
    console.log('Success:', result);
  })
  .catch(error => {
    console.error('Error:', error);
})


}

/* 
fetch('http://localhost:8000/files', {
  method: 'POST',
  body: image,
  headers: { 'Content-Type': 'multipart/form-data',
  'accept' : 'application/json'}
})
*/
setupPage();
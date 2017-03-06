// Load the data
var camData;
Papa.parse("data/camera_and_gaze_smooth.csv", {
            download: true,
            header: true,
            dynamicTyping: true,
            complete: function(results) {
                camData = results;
                startVisualization();
          }
});


//Call init and animation loops once everything has loaded
//$( document ).ready(function(){
//    init();
//    animate();  
//})

function startVisualization(){
    init();
    animate();
}


// Set up the Scene, run init
var scene, camera, renderer;
var container, stats;
var targetTexture = 'textures/target_small.jpg';
var floorTexture = 'textures/floor.jpg';

var targetWidth = 76;
var targetHeight = 82;
var roomWidth = targetWidth*3;
var roomHeight = targetHeight*1.5;
var roomDepth = roomWidth*1.5;

// set up paths
var drawCount;
var path;
var maxLines = 500; 
function init(){

    // set up container
    container = document.createElement('div');
    document.body.appendChild(container);
    
    // create the scene, set size
    scene = new THREE.Scene();
    var WIDTH = window.innerWidth,
        HEIGHT = window.innerHeight;

    
    // create a renderer and add it to the DOM
    renderer = new THREE.WebGLRenderer({antialias:true});
    renderer.setSize(WIDTH, HEIGHT);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    //document.body.appendChild(renderer.domElement);
    
    // create the camera, back it off the origin a bit
    camera = new THREE.PerspectiveCamera(45, WIDTH/HEIGHT, 0.1, 20000);
    camera.lookAt(0,0,0)
    camera.position.set(250,100,300);
    scene.add(camera);
    
    // event listener to deal with window resizing
    window.addEventListener('resize', function(){
        var WIDTH = window.innerWidth,
            HEIGHT = window.innerHeight;
        renderer.setSize(WIDTH, HEIGHT);
        // need to update camera aspect ratio
        camera.aspect = WIDTH/HEIGHT;
        camera.updateProjectionMatrix();
    });
    
    // set BG color
    renderer.setClearColor(0x664F4C, 1);
    
    // Set up lights
    var ambient = new THREE.AmbientLight( 0xffffff, 0.8 );
    scene.add( ambient );

    var spotlight = new THREE.SpotLight( 0xffffff, 0.3 );
    spotlight.position.set(100, 200, 100);
    spotlight.castShadow = true;
    spotlight.shadow.camera.near = 100;
    scene.add(spotlight);
    
    // build Room
    createRoom(floorTexture, roomWidth, roomHeight, roomDepth);
    
    // create Target
    createTarget(targetTexture, targetWidth, targetHeight);
    
    // draw Avatar
    avatar = createAvatar();
    scene.add(avatar);
    
    // add Avatar path
    //path = createPaths();
    maxLines = camData.data.length;
    var geom = new THREE.BufferGeometry();
	var positions = new Float32Array( maxLines * 3 ); // 3 vertices per point
	geom.addAttribute( 'position', new THREE.BufferAttribute( positions, 3 ) );
    drawCount = 2;
    geom.setDrawRange(0,drawCount);
    var mat = new THREE.LineBasicMaterial( { color: 0x167BC1, linewidth: 2 } );
    path = new THREE.Line(geom, mat);
    path.castShadow = true;
    scene.add(path);
    updatePath();
    
    // add Gaze Spot
    gazeSpot = createGazeSpot();
    scene.add(gazeSpot);
    
    // draw the Gaze Line
    //gazeLine = createGazeLine();
    var geom = new THREE.Geometry();
    geom.vertices.push( new THREE.Vector3(0,0,0) );
    geom.vertices.push( new THREE.Vector3(10,10,1) );
    var mat = new THREE.LineBasicMaterial({color: 0xff0000, linewidth: 1})
    gazeLine = new THREE.Line(geom, mat)
    gazeLine.dynamic = true;
    scene.add(gazeLine);
    
    // Add Coordinate Axes to the Scene
    //axes = buildAxes(10);
    //scene.add(axes);
    
    // add Orbit controls to the mouse
    controls = new THREE.OrbitControls(camera, renderer.domElement);  

    // add the Stats monitor
    //stats = new Stats();
    //container.appendChild(stats.dom);
}


// Animate the Scene
var camX, camY, camZ;
var camTheta, camRX, camRY, camRZ;
var gazeDistance;
frameCounter = 0;
var camChange = 1;

function animate(){

    // request the animation frame and set framerate
    setTimeout( function() {
        requestAnimationFrame( animate );
    }, 1000 / 25 );
    
    // convert camera values to this cooridinate system (origin at center of artwork)
    camX = camData.data[frameCounter].camX - targetWidth/2;
    camY = camData.data[frameCounter].camY*-1 + targetHeight/2;
    camZ = camData.data[frameCounter].camZ*-1;
    camTheta = camData.data[frameCounter].camTheta;
    camRX = camData.data[frameCounter].camRX;
    camRY = camData.data[frameCounter].camRY*-1;
    camRZ = camData.data[frameCounter].camRZ*-1;
    
    gazeX = camData.data[frameCounter].obj_gazeX - targetWidth/2;
    gazeY = camData.data[frameCounter].obj_gazeY*-1 + targetHeight/2;
    
    // update Avatar Position & Orientation
    avatar.position.set(camX, camY, camZ);
    var rotQuart = new THREE.Quaternion();
    rotQuart.setFromAxisAngle(new THREE.Vector3(camRX, camRY, camRZ), camTheta);
    avatar.setRotationFromQuaternion(rotQuart);
    
    // update Avatar Path
    path.geometry.setDrawRange(0,frameCounter);
    
    // update Gaze Line and Gaze Spot
    gazeLine.geometry.vertices[0].set(camX, camY, camZ);
    gazeLine.geometry.vertices[1].set(gazeX, gazeY, 1);
    gazeLine.geometry.verticesNeedUpdate = true;
    
    gazeSpot.position.set(gazeX, gazeY, 1);
    gazeDistance = gazeLine.geometry.vertices[0].distanceTo(gazeLine.geometry.vertices[1]);
    var scaleFactor = Math.tan(0.0436) * gazeDistance;   // based on 5deg fovea
    gazeSpot.scale.set(scaleFactor, scaleFactor, 1);

    // update Camera
    if (camera.position.x > 250){
        camChange = -1;
    } else if (camera.position.x < -250){
        camChange = 1;
    }
    camera.position.x += camChange;

    
    // render the scene
    //stats.begin();
    renderer.render(scene, camera);
    controls.update();
    //stats.end();
    
    // update frameCounter
    frameCounter++;
    if (frameCounter == camData.data.length){
        frameCounter = 0;
        
        // reset paths
        updatePath(path);
        path.geometry.attributes.position.needsUpdate=true;
    }
}


//******************** Add'l Functions
// Add Axes to the room for reference
function buildAxes(length){
    var axes = new THREE.Object3D;
    
    axes.add( buildAxis( new THREE.Vector3(0,0,0), new THREE.Vector3(length, 0, 0), 0xFF0000, false) ); // +X
    axes.add( buildAxis( new THREE.Vector3(0,0,0), new THREE.Vector3(0, length, 0), 0x00FF00, false) ); // +Y
    axes.add( buildAxis( new THREE.Vector3(0,0,0), new THREE.Vector3(0, 0, length), 0x0000FF, false) ); // +Z
    
    return axes;
}

function buildAxis(src, dst, colorHex, dashed){
    var geom = new THREE.Geometry(),
        mat;
    
    if(dashed) {
        mat = new THREE.LineDashedMaterial({linewidth:3, color: colorHex, dashSize: 3, gapSize:3 });
    } else {
        mat = new THREE.LineBasicMaterial({linewidth: 3, color: colorHex});
    }
    
    geom.vertices.push(src.clone());
    geom.vertices.push(dst.clone());
    geom.computeLineDistances();
    
    var axis = new THREE.Line( geom, mat, THREE.LineSegments );
    return axis;
}


// Build the Target (i.e. painting)
function createTarget(targetTexure, w, h){
    
    // loader function to load texture and apply to plane
    var loader = new THREE.TextureLoader();
    loader.load(targetTexture, function(texture){
        var geom = new THREE.PlaneGeometry(w, h);
        var mat = new THREE.MeshBasicMaterial({map: texture, side: THREE.DoubleSide});
        var mesh = new THREE.Mesh(geom, mat);
        mesh.position.set(0, 0, 0);
        scene.add(mesh);
    });
}


// Build the Room
function createRoom(floorTexture, w, h, d){
    
    // Build Floor
    var loader = new THREE.TextureLoader();
    loader.load(floorTexture, function(texture){
        var geom = new THREE.PlaneGeometry(w, d);
        var mat = new THREE.MeshLambertMaterial({map: texture, side: THREE.DoubleSide});
        var mesh = new THREE.Mesh(geom, mat);
        mesh.position.set(0, -h/2, d/2);
        mesh.rotation.set(Math.PI/2, 0, 0);
        mesh.receiveShadow = true;
        scene.add(mesh);
    });
    
    // Build Backwall
    var geom = new THREE.PlaneGeometry(w, h);
    var mat = new THREE.MeshBasicMaterial({color: 0xfcfbe6, side: THREE.DoubleSide});
    var mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(0, 0, -1);
    mesh.receiveShadow = true;
    scene.add(mesh);
}


// Create the Avatar
function createAvatar(){
    
    // create head
    var geom = new THREE.BoxGeometry(6,6,6);
    var mat = new THREE.MeshLambertMaterial({color: 0xff0000});
    var mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(0, 0, 55);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    
    return mesh;
}


// Create Gaze Spotlight
function createGazeSpot(){
    var geom = new THREE.CircleGeometry(1,32);
    var mat = new THREE.MeshBasicMaterial({color: 0xff0000, transparent: true, opacity:.55});
    var circle = new THREE.Mesh(geom, mat);
    
    return circle;
}



// Draw Path
function createPaths(){
    maxLines = camData.data.length;
    var geom = new THREE.BufferGeometry();
    var positions = new Float32Array(maxLines*3);
    geom.addAttribute('positions', new THREE.BufferAttribute(positions, 3));
    geom.setDrawRange(0,2);
    var mat = new THREE.LineBasicMaterial({color: 0x000000, linewidth:3});
    
    // line
    line = new THREE.Line(geom, mat);
    return line;

}


function updatePath(){
    var positions = path.geometry.attributes.position.array;
    var x = y = z = index = 0;
    
    for (var i = 0; i < maxLines; i++){
        positions[ index ++ ] = camData.data[i].camX - targetWidth/2;
        positions[ index ++ ] = camData.data[i].camY*-1 + targetHeight/2;
        positions[ index ++ ] = camData.data[i].camZ*-1;
    }

}


import { useEffect, useState, useCallback } from "react";
import { LoadScript, GoogleMap, InfoWindow, HeatmapLayer, Marker } from "@react-google-maps/api";
import GarbageImage from "../images/garbage.png";
import Garbage from "./Garbage";

// Backend
import Modal from "./Modal";
import FileUpload from "./FileUpload";
import { db } from './firebase.js'; 
import { collection, getDocs } from 'firebase/firestore'; 

const containerStyle = {
    width: "100vw",
    height: "100vh"
};

const center = {
    lat: 1.492659,
    lng: 103.7413591
};

const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
const libraries = ["visualization"];

// Custom map styling
const customMapStyle = [
    {
        featureType: "water",
        elementType: "geometry",
        stylers: [{ color: "#003366" }] // Darker blue for the ocean
    },
    {
        featureType: "landscape",
        elementType: "geometry",
        stylers: [{ visibility: "simplified" }, { color: "#f5f5f5" }]
    },
    {
        featureType: "administrative",
        elementType: "geometry.stroke",
        stylers: [{ color: "#000000" }, { weight: 1 }]
    },
    {
        featureType: "road",
        stylers: [{ visibility: "off" }]
    },
    {
        featureType: "poi",
        stylers: [{ visibility: "off" }]
    },
    {
        featureType: "transit",
        stylers: [{ visibility: "off" }]
    }
];

const noLabelsStyle = [
    ...customMapStyle,
    {
        featureType: "all",
        elementType: "labels",
        stylers: [{ visibility: "off" }]
    }
];

function Map() {
    // const [garbageData, setGarbageData] = useState([]); // Using a state variable to store garbage data
    const [isFileUploadVisible, setFileUploadVisible] = useState(false);

    const showFileUpload = () => {
        setFileUploadVisible(true);
    };

    const hideFileUpload = () => {
        setFileUploadVisible(false);
    };

    const garbageData = [
        new Garbage(1.4505, 103.5658, "2024-10-14", "https://3.bp.blogspot.com/-61RRlAbe-oI/WY9cjWUcqWI/AAAAAAAAAhs/1wFrbz13fDAaZYWHqQ6b5s2moW6OE_9zACLcBGAs/s1600/IMG_4980.jpg", 5),
        new Garbage(1.43, 103.5858, "2024-10-13", "", 2),
        new Garbage(1.4605, 103.5598, "2024-10-15", "", 3),
        new Garbage(1.4450, 103.5708, "2024-10-14", "", 1),
        new Garbage(1.4555, 103.5758, "2024-10-16", "", 9),
        new Garbage(1.4375, 103.5808, "2024-10-13", "", 2),
        new Garbage(1.4425, 103.5558, "2024-10-15", "", 3),
        new Garbage(1.4655, 103.5908, "2024-10-14", "", 1),
        new Garbage(1.4250, 103.5658, "2024-10-16", "", 2),
        new Garbage(1.4705, 103.5758, "2024-10-13", "https://firebasestorage.googleapis.com/v0/b/airecondrone.appspot.com/o/DJI_1005.JPG?alt=media&token=8ec93425-6413-4018-8c4f-8962fa10a4a2", 5),
        new Garbage(1.4325, 103.5958, "2024-10-15", "", 9),
        new Garbage(1.4575, 103.5508, "2024-10-14", "", 1)
        // Add more as needed
    ];

    // const fetchGarbageData = async () => {
    //     try {
    //     const querySnapshot = await getDocs(collection(db, 'trash_detection'));
    //     const fetchedGarbageData = querySnapshot.docs.map((doc) => {
    //         const docData = doc.data();
    //         return new Garbage(
    //         docData.latitude,
    //         docData.longitude,
    //         docData.date_created,
    //         docData.image_url,
    //         docData.trash_count
    //         );
    //     });
    //     setGarbageData([...preExistingData, ...fetchedGarbageData]); 
    //     } catch (error) {
    //     console.error('Error fetching documents: ', error);
    //     }
    // };

    // useEffect(() => {
    //     fetchGarbageData();
    // }, []); // Run once on mount

    const [isLoaded, setIsLoaded] = useState(false);
    const [map, setMap] = useState(null);
    const [showGarbage, setShowGarbage] = useState(true);
    const [activeMarker, setActiveMarker] = useState(null);
    const [showHeatmap, setShowHeatmap] = useState(false);
    const [heatmapData, setHeatmapData] = useState([]);
    const [mapType, setMapType] = useState("roadmap");
    const [showLabels, setShowLabels] = useState(true);

    const onLoad = useCallback((map) => {
        setMap(map);
        setIsLoaded(true);
    }, []);

    const onUnmount = useCallback(() => {
        setMap(null);
    }, []);

    useEffect(() => {
        if (map && window.google && showHeatmap && isLoaded) {
            const data = garbageData
            .filter(garbage => garbage.trash_count > 0) // Show heatmap if trash count > 0
            .map(garbage => ({
                location: new window.google.maps.LatLng(garbage.lat, garbage.lng),
                weight: garbage.trash_count
            }));
            setHeatmapData(data);
        } else {
            setHeatmapData([]);
        }
    }, [map, isLoaded, showHeatmap]);

    const toggleLabels = useCallback(() => {
        if (map) {
            setShowLabels(prev => !prev);
            map.setOptions({
                styles: showLabels ? noLabelsStyle : customMapStyle
            });
        }
    }, [map, showLabels]);

    const heatmapOptions = {
        radius: 40,
        opacity: 0.7,
        gradient: [
            'rgba(0, 0, 255, 0)',
            'rgba(0, 255, 255, 1)',
            'rgba(0, 255, 0, 1)',
            'rgba(255, 255, 0, 1)',
            'rgba(255, 0, 0, 1)'
        ]
    };

    const toggleGarbageView = () => setShowGarbage(prev => !prev);
    const toggleHeatmapView = () => setShowHeatmap(prev => !prev);
    const toggleMapType = () => {
        setMapType(prevType => (prevType === "roadmap" ? "satellite" : "roadmap"));
    };

    return (
        <>
            <LoadScript googleMapsApiKey={apiKey} libraries={libraries}>
                <GoogleMap
                    mapContainerStyle={containerStyle}
                    center={center}
                    zoom={9.8}
                    onLoad={onLoad}
                    onUnmount={onUnmount}
                    options={{
                        styles: showLabels ? customMapStyle : noLabelsStyle,
                        disableDefaultUI: true,
                        zoomControl: true,
                        mapTypeControl: false,
                        streetViewControl: false,
                        fullscreenControl: false,
                        mapTypeId: mapType
                    }}
                >
                    {isLoaded && showGarbage && garbageData
                    .filter(garbage => garbage.trash_count > 0) // Show icon if trash count > 0
                    .map((garbage, index) => (
                        <Marker
                            key={index}
                            position={{ lat: garbage.lat, lng: garbage.lng }}
                            onClick={() => setActiveMarker(index)}
                            icon={{
                                url: GarbageImage,
                                scaledSize: new window.google.maps.Size(40, 40) // Adjust size of icon
                            }}
                        />
                    ))}

                    {activeMarker !== null && (
                        <InfoWindow
                            position={{
                                lat: garbageData[activeMarker].lat,
                                lng: garbageData[activeMarker].lng
                            }}
                            onCloseClick={() => setActiveMarker(null)}
                        >
                            <div style={{ color: '#333', fontFamily: 'Arial, sans-serif' }}>
                                <h3 style={{ color: '#1a73e8', marginBottom: '10px' }}>Garbage Information</h3>
                                <p>
                                    <strong style={{ color: '#4285f4' }}>Date & Time: </strong> 
                                    {garbageData[activeMarker].date || "No Date Provided"}
                                    <br></br>
                                    <strong style={{ color: '#4285f4' }}>Number of Trash: </strong> 
                                    {garbageData[activeMarker].trash_count || "0"}
                                </p>
                                {garbageData[activeMarker].picture ? (
                                    <img src={garbageData[activeMarker].picture} alt="Garbage" style={{ maxWidth: '100%', height: 'auto', marginTop: '10px' }} />
                                ) : (
                                    <p style={{ color: '#5f6368', fontStyle: 'italic', marginTop: '10px' }}>No Picture Available</p>
                                )}
                            </div>
                        </InfoWindow>
                    )}

                    {showHeatmap && <HeatmapLayer data={heatmapData} options={heatmapOptions} />}
                </GoogleMap>
            </LoadScript>
            <div style={{
    position: 'absolute',
    bottom: '195px',
    left: '20px',
    backgroundColor: 'white',
    borderRadius: '5px',
    boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
    zIndex: 1000,
}}>
    <button 
        onClick={showFileUpload}
        style={{
            backgroundColor: 'white',
            color: 'black',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer',
        }}
    >
        Upload Image
    </button>

    <Modal isOpen={isFileUploadVisible} onClose={hideFileUpload}>
        <FileUpload onUploadSuccess={() => {
            hideFileUpload(); // Close the modal
            fetchGarbageData(); // Refresh the garbage data
            fetchHeatmapData();
        }} />
    </Modal>
</div>
            <div style={{
    position: 'absolute',
    bottom: '20px',
    left: '20px',
    backgroundColor: 'white',
    padding: '10px',
    borderRadius: '5px',
    boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
    zIndex: 1000,
    display: 'flex',
    flexDirection: 'column',  // Stack buttons vertically
    gap: '10px'               // Add spacing between buttons
}}>
    <button 
        onClick={toggleGarbageView}
        style={{
            padding: '5px 10px',
            backgroundColor: showGarbage ? '#4CAF50' : '#f44336',
            color: 'white',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer'
        }}
    >
        {showGarbage ? 'Hide' : 'Show'} Markers
    </button>
    <button 
        onClick={toggleHeatmapView}
        style={{
            padding: '5px 10px',
            backgroundColor: showHeatmap ? '#4CAF50' : '#f44336',
            color: 'white',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer'
        }}
    >
        {showHeatmap ? 'Hide' : 'Show'} Heatmap
    </button>
    <button 
        onClick={toggleMapType}
        style={{
            padding: '5px 10px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer'
        }}
    >
        Switch to {mapType === "roadmap" ? "Satellite" : "Roadmap"} View
    </button>
    <button 
        onClick={toggleLabels}
        style={{
            padding: '5px 10px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer'
        }}
    >
        Toggle Labels
    </button>
</div>

        </>
    );
}

export default Map;
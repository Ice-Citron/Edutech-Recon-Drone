import { useEffect, useState, useCallback } from "react";
import { LoadScript, GoogleMap, InfoWindow, HeatmapLayer, Marker } from "@react-google-maps/api";
import GarbageImage from "../images/garbage.png";
import Garbage from "./Garbage";

// Backend
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

const libraries = ["visualization"];

function Map() {
    const [garbageData, setGarbageData] = useState([]); // Using a state variable to store garbage data

    const fetchGarbageData = async () => {
        try {
        const querySnapshot = await getDocs(collection(db, 'trash_detection'));
        const fetchedGarbageData = querySnapshot.docs.map((doc) => {
            const docData = doc.data();
            return new Garbage(
            docData.latitude,
            docData.longitude,
            docData.date_created,
            docData.image_url
            );
        });
        setGarbageData(fetchedGarbageData); // Update the state with fetched data
        } catch (error) {
        console.error('Error fetching documents: ', error);
        }
    };

    useEffect(() => {
        fetchGarbageData();
    }, []); // Run once on mount

    const [showGarbage, setShowGarbage] = useState(false);
    const [showHeatmap, setShowHeatmap] = useState(false);
    const [isLoaded, setIsLoaded] = useState(false);
    const [activeMarker, setActiveMarker] = useState(null);
    const [heatmapData, setHeatmapData] = useState([]);
    const [map, setMap] = useState(null);
    const [mapType, setMapType] = useState("roadmap");
    const [showLabels, setShowLabels] = useState(true);

    const toggleLabels = () => {
        if (map) {
            console.log("Toggling labels. Current state:", showLabels);
            const newStyle = showLabels
                ? [
                    ...customMapStyle,
                    { featureType: "administrative", elementType: "labels", stylers: [{ visibility: "off" }] },
                    { featureType: "road", elementType: "labels", stylers: [{ visibility: "off" }] },
                    { featureType: "poi", elementType: "labels", stylers: [{ visibility: "off" }] },
                    { featureType: "transit", elementType: "labels", stylers: [{ visibility: "off" }] },
                    { featureType: "water", elementType: "labels", stylers: [{ visibility: "off" }] }
                ]
                : [...customMapStyle];
    
            console.log("New style being applied:", newStyle);
            
            map.setOptions({ styles: [] });
            setTimeout(() => {
                map.setOptions({ styles: newStyle });
            }, 100);
    
            setShowLabels(prev => !prev);
        }
    };

    const onLoad = useCallback((map) => {
        setMap(map);
        setIsLoaded(true);
        console.log('Map loaded:', map);
    }, []);

    const onUnmount = useCallback(() => {
        setMap(null);
    }, []);


    useEffect(() => {
        if (map && window.google && showHeatmap && isLoaded) {
            const data = garbageData.map(garbage => ({
                location: new window.google.maps.LatLng(garbage.lat, garbage.lng),
                weight: 1
            }));
            setHeatmapData(data);
        } else {
            setHeatmapData([]);
        }
    }, [map, isLoaded, showHeatmap]);

    const heatmapOptions = {
        radius: 40,
        opacity: 0.8,
        gradient: [
            'rgba(0, 255, 0, 0)',    
            'rgba(0, 255, 0, 1)',
            'rgba(255, 255, 0, 1)',
            'rgba(255, 128, 0, 1)',
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
                    zoom={9.9}
                    onLoad={onLoad}
                    onUnmount={onUnmount}
                    options={{
                        styles: customMapStyle,
                        disableDefaultUI: true,
                        zoomControl: true,
                        mapTypeControl: false,
                        streetViewControl: false,
                        fullscreenControl: false,
                        mapTypeId: mapType
                    }}
                >
                    {isLoaded && showGarbage && garbageData.map((garbage, index) => (
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
                                <p><strong style={{ color: '#4285f4' }}>Date:</strong> {garbageData[activeMarker].date || "No Date Provided"}</p>
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
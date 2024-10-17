// https://www.youtube.com/watch?v=Hd08pIzJiHc

import { getFirestore } from 'firebase-admin/firestore';
import { initializeApp, getApps, cert } from 'firebase-admin/app';
import express from 'express';

const app = express();
const PORT = process.env.PORT || 3000;

const apps = getApps();

if (!apps.length) {
    initializeApp({
        credential: cert('airecondrone-firebase-adminsdk-1oiv0-cfdb4ae33a.json')
    });
}

app.get('/predictions', async (req, res) => {
    try {
        const db = getFirestore();
        const predictionsSnap = await db.collection('predictions').get();
        const predictionsData = predictionsSnap.docs.map(doc => ({
            uuid: doc.id,
            ...doc.data()
        }));
        res.json(predictionsData); 
    } catch (error) {
        console.error("Error fetching images:", error);
        res.status(500).send("Error fetching images");
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});

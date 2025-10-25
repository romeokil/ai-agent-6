import * as dotenv from 'dotenv'
dotenv.config();
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';

// Phase-1 hai ye process esko ek hi baar krna hai 
// eske baad Phase-2 jo hai usko baar baar krna hoga qki 
// user input toh baar baar de skta hai.

// Step-1 sbse pehle pdf file ko load krege.
const PDF_PATH='./Dsa.pdf';
const Pdfloader= new PDFLoader(PDF_PATH);
const rawDocs=await Pdfloader.load();

// console.log(JSON.stringify(rawDocs,null,2));
// console.log(rawDocs.length)

// Step-2 ab hmlog es pdf document ko chunks me divide kr dege taaki
// hmlog ko pura document ko ni dena pde or hmlog chunkSize mtlb kitna 
// division krna chahte hai hmlog chunkSize me likhte hai
// or overlapping krna chahiye taaki ho skta hai jo doondh rhe hai wo khi
// dusra wala segment me ho. 1-1000, 800-1800, 1600-2600 aise aise chunks divide hue
// or overlap 200 ka daale toh.

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
const chunkedDocs = await textSplitter.splitDocuments(rawDocs);

console.log(JSON.stringify(chunkedDocs.slice(0, 2), null, 2));


// Step-3-> ab hmlog chunks create kr liye hai ab baari hai
// ki hmlog usko vector me convert kre toh iske liye hmlog ke pass
// embedding models hote hai jiske help se hmlog aaram se convert kr
// skte hai chunks ko vector me.

// ye bs initialize kr rhe hai embedding ko and hmlog 
// gemini-ai ka embedding ka use kr rhe hai.

const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GOOGLE_API_KEY,
    model: 'text-embedding-004',
  });


// Step-4-> Pinecone client ko initialize kr rhe hai

const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

// Step-5-> Hmlog chunks ko embedded model me daal rhe hai
// or pinecone database me vector convert hone ke baad usme bhi insert kr de rhe hai.
// or ye dono chiz hmlog ek hi time pe kr rhe hai.

await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
    pineconeIndex,
    maxConcurrency: 5,
  });




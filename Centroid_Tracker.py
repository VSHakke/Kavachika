from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        print("CentroidTracker started")
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        """Register a new object with the given centroid."""
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        """Deregister an object by its ID."""
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        """Update the tracker with the current bounding box rectangles."""
        if len(rects) == 0:
            # If no rectangles, increment the disappeared count for each object
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # Deregister if the object has disappeared for too long
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # Calculate the centroids for each rectangle
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            # If no existing objects, register each centroid
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # Existing objects, perform the assignment
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Compute the distance between each pair of object centroids and input centroids
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # Find the rows and columns that need to be matched
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            # Handle any objects that were not matched
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

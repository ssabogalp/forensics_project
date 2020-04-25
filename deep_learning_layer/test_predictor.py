from evidence_predictor import EvidencePredictor

#To use this unite test you need to change the import as non package in evidence_predictor
c=EvidencePredictor()
c.train(2,10,"data/abuse.csv")
#this should be caught as evidence
print(c.predict("lets abuse her",10))
function op = trainKNN( data, train, test )

    trdata = data(train,:);
    tedata = data(test,1:43);
    
    [ model, ~ ] = trainClassifier( trdata );
    op = model.predictFcn( tedata );

end
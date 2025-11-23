# Guide de Contribution

Merci de votre intérêt pour contribuer au projet Brain! Ce document fournit des lignes directrices pour contribuer au projet.

## Code de Conduite

En participant à ce projet, vous acceptez de respecter notre code de conduite.

## Comment Contribuer

### Rapporter des Bugs

Si vous trouvez un bug, veuillez créer une issue avec:
- Une description claire du problème
- Les étapes pour reproduire le bug
- Le comportement attendu vs le comportement actuel
- Votre environnement (OS, version Python, etc.)

### Proposer des Fonctionnalités

Pour proposer une nouvelle fonctionnalité:
1. Vérifiez qu'elle n'existe pas déjà dans les issues
2. Créez une issue décrivant la fonctionnalité
3. Expliquez pourquoi cette fonctionnalité serait utile
4. Attendez les retours de la communauté

### Pull Requests

1. Forkez le projet
2. Créez votre branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add AmazingFeature'`)
4. Poussez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

#### Standards de Code

- Suivre PEP 8
- Utiliser Black pour le formatage
- Ajouter des docstrings pour toutes les fonctions
- Écrire des tests pour le nouveau code
- Maintenir une couverture de code > 70%

#### Tests

Assurez-vous que tous les tests passent:
```bash
make test
```

#### Documentation

- Documentez les nouvelles fonctionnalités
- Mettez à jour le README si nécessaire
- Ajoutez des exemples d'utilisation

## Style de Code

### Python

- Suivre PEP 8
- Ligne maximale de 127 caractères
- Utiliser des noms descriptifs pour les variables
- Ajouter des commentaires pour la logique complexe

### Commits

Format des messages de commit:
```
type(scope): description courte

Description détaillée si nécessaire

Fixes #123
```

Types:
- `feat`: Nouvelle fonctionnalité
- `fix`: Correction de bug
- `docs`: Documentation
- `style`: Formatage
- `refactor`: Refactoring
- `test`: Ajout de tests
- `chore`: Maintenance

## Questions

Pour toute question, ouvrez une issue ou contactez l'équipe.

Merci de contribuer! 🎉
